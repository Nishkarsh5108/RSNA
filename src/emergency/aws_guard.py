"""AWS GPU guard for the RSNA sprint: check the account, launch ONE GPU instance with three kill-switch layers attached
at birth, report the hourly burn, and tear everything down.

Credentials: boto3 reads C:/Users/nishk/.aws/credentials.txt through AWS_SHARED_CREDENTIALS_FILE; nothing is printed or
copied. Local state (instance, alarm, budget action, key, security group) goes to D:/rsna_work/aws/state.json.

Kill-switch layers (attached right after run-instances returns; if layer 1B or 3 cannot be attached, the instance is
terminated at once):
  1B  AWS Budgets action RUN_SSM_DOCUMENTS / STOP_EC2_INSTANCES (the console's "stop EC2 instances") on this instance at 100 % of RSNA-Hard-Cap-80USD, run by
      AWSBudgetsEC2KillSwitchRole. Budget data refreshes only a few times a day, so this layer is slow; 2 and 3 are fast.
      (The account's existing action only denies the agent user all IAM permissions; it does not stop instances.)
  2   On the instance (user data): `shutdown -h +<minutes>` hard wall clock, and a GPU-idle watchdog (< 5 % for 25
      consecutive minutes after a 15-minute grace). Shutdown terminates the instance (InstanceInitiatedShutdownBehavior =
      terminate) and the root gp3 volume has DeleteOnTermination = true, so nothing is left to bill.
      Upload keep-alive: while /var/tmp/KEEPALIVE is fresher than 10 min (the uploader touches it), the idle counter
      does not advance and a light CPU load (about 8 % of 4 vCPUs) keeps the layer-3 alarm quiet. A dead uploader stops
      touching it, so both watchdogs resume within 10 minutes.
  3   CloudWatch alarm AutoTerminate-Idle-<id>: average CPUUtilization < 3 % for 6 x 5 min -> ec2:terminate.

Usage (python from D:/rsna_work/aws_venv):
  aws_guard.py check                                  account, budgets, actions, G/VT quotas, month-to-date cost
  aws_guard.py selftest --region R                    create + delete a test alarm and a test budget action (no instance)
  aws_guard.py launch --region R --type g5.xlarge [--minutes 420] [--root-gb 80] [--dry-run]
  aws_guard.py status                                 running instances, unattached volumes, alarms, actions, burn
  aws_guard.py teardown [--region R --instance i-...] terminate + delete alarm, budget action, key pair, security group
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ["AWS_SHARED_CREDENTIALS_FILE"] = r"C:\Users\nishk\.aws\credentials.txt"
os.environ.pop("AWS_PROFILE", None)
import boto3  # noqa: E402
import botocore  # noqa: E402

REGIONS = ("us-east-1", "us-west-2", "ap-south-1")
BUDGET = "RSNA-Hard-Cap-80USD"
ROLE_NAME = "AWSBudgetsEC2KillSwitchRole"
QUOTA_ON_DEMAND, QUOTA_SPOT = "L-DB2E81BA", "L-3819A6DF"
AMI_NAME = "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*"
STATE_DIR = "D:/rsna_work/aws"
STATE = f"{STATE_DIR}/state.json"
# On-demand USD/hour, estimates for the burn report only (the pricing API is not permitted for this user)
PRICE_EST = {"g5.xlarge": 1.006, "g6.xlarge": 0.805, "g5.2xlarge": 1.212, "g6.2xlarge": 0.978}
GP3_USD_GB_MONTH = 0.08

USER_DATA = r"""#!/bin/bash
# Layer 2a: hard wall-clock ceiling; shutdown terminates this instance (InstanceInitiatedShutdownBehavior=terminate)
shutdown -h +{minutes} "Hard {minutes}-minute wall-clock safety limit reached; terminating instance."
# Layer 2b: GPU-idle watchdog: < 5 % for 25 consecutive checks after a 15-minute grace -> terminate.
# A fresh /var/tmp/KEEPALIVE (< 10 min, touched by the uploader) pauses the count.
(
  sleep 900
  idle_count=0
  while true; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{{s+=$1}} END {{print s+0}}')
    if [ -n "$(find /var/tmp/KEEPALIVE -mmin -10 2>/dev/null)" ]; then
      idle_count=0
    elif [ "${{util:-0}}" -lt 5 ]; then
      idle_count=$((idle_count + 1))
    else
      idle_count=0
    fi
    if [ "$idle_count" -ge 25 ]; then
      shutdown -h now "GPU idle (<5%) for 25 minutes; auto-terminating to save credits."
      exit 0
    fi
    sleep 60
  done
) &
# Upload keep-alive for the layer-3 CPU alarm: about 20 s of one busy core per minute while KEEPALIVE is fresh
(
  while true; do
    if [ -n "$(find /var/tmp/KEEPALIVE -mmin -10 2>/dev/null)" ]; then
      timeout 20 nice -n 19 sh -c 'while :; do :; done'
    fi
    sleep 40
  done
) &
"""


def session(region="us-east-1"):
    return boto3.Session(region_name=region)


def account():
    return session().client("sts").get_caller_identity()["Account"]


def err(e):
    return f"{e.response['Error']['Code']}: {e.response['Error'].get('Message', '')[:200]}"


def existing_action():
    b = session().client("budgets")
    acts = b.describe_budget_actions_for_budget(AccountId=account(), BudgetName=BUDGET)["Actions"]
    return b, acts


def role_arn_and_subscribers():
    _, acts = existing_action()
    for a in acts:
        if a["ExecutionRoleArn"].endswith("/" + ROLE_NAME):
            return a["ExecutionRoleArn"], a["Subscribers"]
    raise SystemExit(f"no existing action on {BUDGET} uses {ROLE_NAME}; cannot copy its role and subscribers")


def quota(region, code):
    sq = session(region).client("service-quotas")
    return sq.get_service_quota(ServiceCode="ec2", QuotaCode=code)["Quota"]["Value"]


def cmd_check(_):
    s = session()
    print("identity:", s.client("sts").get_caller_identity()["Arn"].split(":")[-1])
    b, acts = existing_action()
    for x in b.describe_budgets(AccountId=account())["Budgets"]:
        sp = x.get("CalculatedSpend", {})
        print(f"budget {x['BudgetName']}: limit {x['BudgetLimit']['Amount']} actual {sp.get('ActualSpend', {}).get('Amount')}")
    for a in acts:
        print(f"  action {a['ActionType']} at {a['ActionThreshold']['ActionThresholdValue']}% status {a['Status']}")
    for r in REGIONS:
        try:
            print(f"{r}: G/VT on-demand vCPU {quota(r, QUOTA_ON_DEMAND)}, spot vCPU {quota(r, QUOTA_SPOT)}")
        except botocore.exceptions.ClientError as e:
            print(r, "quota:", err(e))
    today = datetime.date.today()
    try:
        ce = s.client("ce").get_cost_and_usage(TimePeriod={"Start": today.replace(day=1).isoformat(),
                                                           "End": (today + datetime.timedelta(days=1)).isoformat()},
                                                Granularity="MONTHLY", Metrics=["UnblendedCost"])
        print("month-to-date cost:", ce["ResultsByTime"][0]["Total"]["UnblendedCost"])
    except botocore.exceptions.ClientError as e:
        print("cost explorer:", err(e))


def put_alarm(region, iid):
    name = f"AutoTerminate-Idle-{iid}"
    session(region).client("cloudwatch").put_metric_alarm(
        AlarmName=name, MetricName="CPUUtilization", Namespace="AWS/EC2", Statistic="Average", Period=300,
        Threshold=3.0, ComparisonOperator="LessThanThreshold", Dimensions=[{"Name": "InstanceId", "Value": iid}],
        EvaluationPeriods=6, AlarmActions=[f"arn:aws:automate:{region}:ec2:terminate"])
    return name


def put_budget_stop(region, iid):
    role, subs = role_arn_and_subscribers()
    b = session().client("budgets")
    r = b.create_budget_action(
        AccountId=account(), BudgetName=BUDGET, NotificationType="ACTUAL", ActionType="RUN_SSM_DOCUMENTS",
        ActionThreshold={"ActionThresholdValue": 100.0, "ActionThresholdType": "PERCENTAGE"},
        Definition={"SsmActionDefinition": {"ActionSubType": "STOP_EC2_INSTANCES", "Region": region, "InstanceIds": [iid]}},
        ExecutionRoleArn=role, ApprovalModel="AUTOMATIC", Subscribers=subs)
    return r["ActionId"]


def delete_budget_action(action_id):
    """The agent user may lack budgets:DeleteBudgetAction; then the owner deletes it in the Budgets console. A leftover
    stop action is harmless once its instance is terminated (it can only stop that instance id)."""
    try:
        session().client("budgets").delete_budget_action(AccountId=account(), BudgetName=BUDGET, ActionId=action_id)
        return True
    except botocore.exceptions.ClientError as e:
        print(f"  could not delete budget action {action_id} ({err(e)[:60]}); owner: delete it in the Budgets console")
        return False


def cmd_selftest(a):
    """Permission test of layers 1B and 3 on a placeholder instance id; both objects are deleted again."""
    fake = "i-0000000000000000f"
    try:
        name = put_alarm(a.region, fake)
        session(a.region).client("cloudwatch").delete_alarms(AlarmNames=[name])
        print("layer 3 (CloudWatch terminate alarm): create + delete OK")
    except botocore.exceptions.ClientError as e:
        print("layer 3 FAILED:", err(e))
    try:
        aid = put_budget_stop(a.region, fake)
        print("layer 1B (budget EC2 stop action): create OK" + (" + delete OK" if delete_budget_action(aid) else ""))
    except botocore.exceptions.ClientError as e:
        print("layer 1B FAILED:", err(e))


def latest_dlami(ec2):
    imgs = ec2.describe_images(Owners=["amazon"], Filters=[
        {"Name": "name", "Values": [AMI_NAME]}, {"Name": "state", "Values": ["available"]},
        {"Name": "architecture", "Values": ["x86_64"]}])["Images"]
    imgs = [i for i in imgs if not i.get("ProductCodes")]           # never a paid Marketplace image
    if not imgs:
        raise SystemExit("no free official DLAMI found")
    return max(imgs, key=lambda i: i["CreationDate"])


def my_ip():
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=20).read().decode().strip()


def save_state(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=1)


def cmd_launch(a):
    ec2 = session(a.region).client("ec2")
    vcpu = ec2.describe_instance_types(InstanceTypes=[a.type])["InstanceTypes"][0]["VCpuInfo"]["DefaultVCpus"]
    q = quota(a.region, QUOTA_ON_DEMAND)
    if vcpu > q:
        if not a.dry_run:
            raise SystemExit(f"{a.type} needs {vcpu} vCPU; the on-demand G/VT quota in {a.region} is {q}")
        print(f"(dry run) {a.type} needs {vcpu} vCPU; quota in {a.region} is {q}: a real launch would be refused")
    if os.path.exists(STATE) and json.load(open(STATE)).get("instance") and not a.dry_run:
        raise SystemExit(f"{STATE} lists a live instance; run teardown first (one instance at a time)")
    ami = latest_dlami(ec2)
    root = ami["RootDeviceName"]
    print(f"AMI {ami['ImageId']} {ami['Name']} (no product codes)")
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    ud = USER_DATA.format(minutes=int(a.minutes))
    spec = dict(ImageId=ami["ImageId"], InstanceType=a.type, MinCount=1, MaxCount=1,
                InstanceInitiatedShutdownBehavior="terminate", UserData=ud,
                BlockDeviceMappings=[{"DeviceName": root, "Ebs": {"VolumeSize": int(a.root_gb), "VolumeType": "gp3",
                                                                   "DeleteOnTermination": True}}],
                MetadataOptions={"HttpTokens": "required"},
                TagSpecifications=[{"ResourceType": t, "Tags": [{"Key": "Name", "Value": f"rsna-gpu-{ts}"},
                                                                {"Key": "Project", "Value": "rsna-knee"}]}
                                   for t in ("instance", "volume")])
    if a.dry_run:
        try:
            ec2.run_instances(DryRun=True, **spec)
        except botocore.exceptions.ClientError as e:
            print("dry run:", err(e))
        return
    key = f"rsna-gpu-{ts}"
    os.makedirs(f"{STATE_DIR}/keys", exist_ok=True)
    pem = f"{STATE_DIR}/keys/{key}.pem"
    material = ec2.create_key_pair(KeyName=key, KeyType="ed25519")["KeyMaterial"]
    with open(pem, "w", newline="\n") as f:
        f.write(material)
    del material
    user = os.environ.get("USERNAME", "")
    subprocess.run(["icacls", pem, "/inheritance:r", "/grant:r", f"{user}:R"], capture_output=True)
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    sg = ec2.create_security_group(GroupName=f"rsna-ssh-{ts}", Description="RSNA sprint: SSH from the owner's IP only",
                                   VpcId=vpc)["GroupId"]
    ip = my_ip()
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                                                                     "IpRanges": [{"CidrIp": f"{ip}/32"}]}])
    st = {"region": a.region, "type": a.type, "key": key, "pem": pem, "sg": sg, "launched": ts, "minutes": int(a.minutes)}
    save_state(st)
    r = ec2.run_instances(KeyName=key, SecurityGroupIds=[sg], **spec)
    iid = r["Instances"][0]["InstanceId"]
    st["instance"] = iid
    save_state(st)
    print("launched", iid, "- attaching layers 3 and 1B")
    try:
        st["alarm"] = put_alarm(a.region, iid)
        save_state(st)
        st["budget_action"] = put_budget_stop(a.region, iid)
        save_state(st)
    except Exception as e:                                             # noqa: BLE001 - any failure means no instance
        print("KILL-SWITCH ATTACH FAILED -> terminating", iid, ":", e)
        ec2.terminate_instances(InstanceIds=[iid])
        raise SystemExit(1)
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    inst = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    beh = ec2.describe_instance_attribute(InstanceId=iid, Attribute="instanceInitiatedShutdownBehavior")
    dot = [m["Ebs"]["DeleteOnTermination"] for m in inst["BlockDeviceMappings"]]
    alarm_ok = bool(session(a.region).client("cloudwatch").describe_alarms(AlarmNames=[st["alarm"]])["MetricAlarms"])
    print(f"verified: shutdown behaviour {beh['InstanceInitiatedShutdownBehavior']['Value']}, DeleteOnTermination {dot},"
          f" alarm {'present' if alarm_ok else 'MISSING'}, budget action {st['budget_action']}")
    if beh["InstanceInitiatedShutdownBehavior"]["Value"] != "terminate" or not all(dot) or not alarm_ok:
        print("verification failed -> terminating")
        ec2.terminate_instances(InstanceIds=[iid])
        raise SystemExit(1)
    st["public_dns"] = inst.get("PublicDnsName")
    save_state(st)
    print(f"ready: ubuntu@{st['public_dns']} (key file {pem}); hard stop after {a.minutes} min;"
          f" estimated {PRICE_EST.get(a.type, float('nan')):.3f} USD/h")


def cmd_status(_):
    burn = 0.0
    for r in REGIONS:
        s = session(r)
        ec2, cw = s.client("ec2"), s.client("cloudwatch")
        for res in ec2.describe_instances(Filters=[{"Name": "instance-state-name",
                                                    "Values": ["pending", "running", "stopping", "stopped"]}])["Reservations"]:
            for i in res["Instances"]:
                hrs = (datetime.datetime.now(datetime.timezone.utc) - i["LaunchTime"]).total_seconds() / 3600
                p = PRICE_EST.get(i["InstanceType"], float("nan")) if i["State"]["Name"] in ("pending", "running") else 0.0
                burn += p
                print(f"{r}: {i['InstanceId']} {i['InstanceType']} {i['State']['Name']} up {hrs:.2f} h, ~{p:.3f} USD/h")
        for v in ec2.describe_volumes()["Volumes"]:
            tag = "UNATTACHED (billing)" if v["State"] == "available" else v["State"]
            print(f"{r}: volume {v['VolumeId']} {v['Size']} GB {tag} ~{v['Size'] * GP3_USD_GB_MONTH / 30:.2f} USD/day")
        for al in cw.describe_alarms(AlarmNamePrefix="AutoTerminate-Idle-")["MetricAlarms"]:
            print(f"{r}: alarm {al['AlarmName']} state {al['StateValue']}")
        for ad in ec2.describe_addresses()["Addresses"]:
            print(f"{r}: elastic IP {ad.get('PublicIp')} {'UNASSOCIATED (billing)' if not ad.get('AssociationId') else ''}")
    _, acts = existing_action()
    for a in acts:
        if a["ActionType"] == "RUN_SSM_DOCUMENTS":
            print("budget stop action", a["ActionId"], a["Definition"]["SsmActionDefinition"]["InstanceIds"], a["Status"])
    print(f"current burn ~{burn:.3f} USD/h (instances only; estimates)")


def cmd_teardown(a):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    region = a.region or st.get("region")
    iid = a.instance or st.get("instance")
    if not region:
        raise SystemExit("no region given and no state file")
    ec2 = session(region).client("ec2")
    if iid:
        ec2.terminate_instances(InstanceIds=[iid])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[iid])
        print("terminated", iid)
        alarm = st.get("alarm") or f"AutoTerminate-Idle-{iid}"
        session(region).client("cloudwatch").delete_alarms(AlarmNames=[alarm])
        print("deleted alarm", alarm)
    _, acts = existing_action()
    for act in acts:
        if act["ActionType"] == "RUN_SSM_DOCUMENTS" and (
                act["ActionId"] == st.get("budget_action") or iid in act["Definition"]["SsmActionDefinition"]["InstanceIds"]):
            if delete_budget_action(act["ActionId"]):
                print("deleted budget action", act["ActionId"])
    if st.get("key"):
        ec2.delete_key_pair(KeyName=st["key"])
        if st.get("pem") and os.path.exists(st["pem"]):
            subprocess.run(["icacls", st["pem"], "/grant:r", f"{os.environ.get('USERNAME', '')}:F"], capture_output=True)
            os.remove(st["pem"])
        print("deleted key pair", st["key"])
    if st.get("sg"):
        for _ in range(10):
            try:
                ec2.delete_security_group(GroupId=st["sg"])
                print("deleted security group", st["sg"])
                break
            except botocore.exceptions.ClientError:
                time.sleep(15)
    if os.path.exists(STATE):
        os.replace(STATE, STATE + f".done_{datetime.datetime.now():%Y%m%d%H%M}")
    cmd_status(a)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    t = sub.add_parser("selftest")
    t.add_argument("--region", default="us-east-1")
    lp = sub.add_parser("launch")
    lp.add_argument("--region", required=True)
    lp.add_argument("--type", default="g5.xlarge")
    lp.add_argument("--minutes", type=int, default=420)
    lp.add_argument("--root-gb", type=int, default=80)
    lp.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    td = sub.add_parser("teardown")
    td.add_argument("--region")
    td.add_argument("--instance")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"check": cmd_check, "selftest": cmd_selftest, "launch": cmd_launch, "status": cmd_status,
     "teardown": cmd_teardown}[a.cmd](a)


if __name__ == "__main__":
    main()

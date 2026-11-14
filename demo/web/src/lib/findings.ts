// The twelve findings, in the models' output order, with plain-language names and the training-label prevalence
// (share of the 4,349 labelled training studies whose fused label is >= 0.5; 1st- and 7th-place label sets averaged).
export interface Finding {
  key: string;
  short: string;
  name: string;
  region: string;
  prevalence: number;
}

export const FINDINGS: Finding[] = [
  { key: "ACL", short: "ACL", name: "ACL tear", region: "Anterior cruciate ligament, intercondylar notch", prevalence: 0.115 },
  { key: "MCL", short: "MCL", name: "MCL injury", region: "Medial collateral ligament", prevalence: 0.021 },
  { key: "Medial Meniscus", short: "MM", name: "Medial meniscus tear", region: "Medial meniscus", prevalence: 0.377 },
  { key: "Lateral Meniscus", short: "LM", name: "Lateral meniscus tear", region: "Lateral meniscus", prevalence: 0.132 },
  { key: "Medial OA", short: "MOA", name: "Medial osteoarthritis", region: "Medial tibiofemoral compartment", prevalence: 0.178 },
  { key: "Lateral OA", short: "LOA", name: "Lateral osteoarthritis", region: "Lateral tibiofemoral compartment", prevalence: 0.062 },
  { key: "PF OA", short: "PFOA", name: "Patellofemoral osteoarthritis", region: "Patellofemoral joint", prevalence: 0.207 },
  { key: "Effusion", short: "EFF", name: "Joint effusion", region: "Suprapatellar recess and joint space", prevalence: 0.328 },
  { key: "Synovitis", short: "SYN", name: "Synovitis", region: "Synovium, Hoffa's fat pad", prevalence: 0.157 },
  { key: "Baker's", short: "BAK", name: "Baker's cyst", region: "Posteromedial popliteal fossa", prevalence: 0.137 },
  { key: "Contusion", short: "CON", name: "Bone contusion", region: "Femoral condyles, tibial plateau, patella", prevalence: 0.12 },
  { key: "Fracture", short: "FX", name: "Fracture", region: "Distal femur, proximal tibia and fibula, patella", prevalence: 0.05 },
];

export const PROJECT = {
  name: "Knee Abnormality Detection",
  short: "KAD",
  institution: "Department of Electronics and Communication, University of Delhi",
};

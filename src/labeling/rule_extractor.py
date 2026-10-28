"""Transparent multilingual rule-based assertion extractor (ConText/NegEx style).

For each report and each of the 12 targets it returns an assertion
  present | uncertain | historical | absent | not_mentioned
plus the clause used as evidence. Developed on non-gold reports only.

Pipeline: Unicode normalisation (incl. the Greek micro-sign fix) -> lower-case,
accent folding (Latin/Greek; Cyrillic kept) -> sentence split -> removal of
history / clinical-question sentences -> clause split -> concept detection
(anatomy + condition, or a standalone condition term) -> negation (pre- and
post-cue, because Turkish/Greek/Croatian/Bulgarian often negate after the noun),
uncertainty and post-operative/historical cues -> per-target aggregation with
priority present > uncertain > historical > absent.
"""
import re
import unicodedata

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RULESET_VERSION = "rules-v1.1"

GREEK_FIXES = {"µ": "μ", "∆": "Δ", "Ω": "Ω"}


def fold(text):
    text = unicodedata.normalize("NFC", str(text))
    for a, b in GREEK_FIXES.items():
        text = text.replace(a, b)
    text = text.replace("ß", "ss").replace("ı", "i").replace("İ", "i").replace("œ", "oe").replace("Œ", "oe")
    # join hard-wrapped lines (a newline not preceded by sentence punctuation)
    text = re.sub(r"(?<![.:;!?])[ \t]*\n[ \t]*", " ", text)
    text = text.lower()
    # strip combining accents (keeps Cyrillic letters, which are not decomposed except й/ё)
    nfd = unicodedata.normalize("NFD", text)
    out = []
    for ch in nfd:
        if unicodedata.category(ch) == "Mn" and not ("Ѐ" <= (out[-1] if out else " ") <= "ӿ"):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def rx(*alts):
    return re.compile("(?:" + "|".join(alts) + ")")


# ------------------------------------------------------------------ anatomy
ACL = rx(r"anterior cruciate", r"\bacl\b", r"cruzado anterior", r"\blca\b", r"on capraz", r"\bocb\b", r"\bacb\b",
         r"prednj\w* krizn", r"\bpkl\b", r"\bplc\b", r"προσθι\w* χιαστ", r"\bπχσ\b", r"vorder\w* kreuzband", r"\bvkb\b",
         r"предн\w* кръстн", r"\bпкв\b", r"\bпкл\b", r"voorste kruisband", r"croise anterieur", r"lca\b", r"\bлка\b")
MCL = rx(r"medial collateral", r"tibial collateral", r"\bmcl\b", r"colateral (medial|interno)", r"\blcm\b", r"\blli\b",
         r"medial kollateral", r"\bmkl\b", r"ic yan bag", r"medijaln\w* kolateraln", r"medijaln\w* postran", r"unutarnj\w* (kolateraln|postran)",
         r"εσω πλαγι\w* συνδεσμ", r"\bεπσ\b", r"innenband", r"medial\w* kollateralband", r"медиалн\w* колатерал",
         r"вътрешн\w* (колатерал|странич)", r"mediale collaterale", r"binnenband", r"collateral medial", r"lateral interne")
CRUCIATES = rx(r"cruciate ligaments", r"cruciates", r"ligamentos cruzados", r"capraz bag", r"krizni ligament", r"kriznih ligament",
               r"χιαστοι συνδεσμοι", r"χιαστων συνδεσμων", r"кръстни връзки", r"kreuzbander", r"kruisbanden", r"ligaments croises")
COLLATERALS = rx(r"collateral ligaments", r"ligamentos colaterales", r"yan bag", r"kollateral bag", r"kolateraln\w* ligament",
                 r"πλαγιοι συνδεσμοι", r"πλαγιων συνδεσμων", r"колатерални връзки", r"seitenband", r"kollateralband\w*er",
                 r"collaterale banden", r"laterale banden", r"ligaments collateraux")
MENISC = rx(r"menisc(?!o-?femor|o-?tibia|o-?capsul|o-?kapsul)", r"menisk(?!o-?femor|o-?tibia)", r"μηνισκ", r"мениск", r"menisque", r"meniskus")
MEDIAL = rx(r"\bmed(i|iy|y|ij)al", r"\binterno", r"\binterne\b", r"\bic\b", r"unutarnj", r"\bεσω\b", r"\binnen",
            r"медиал", r"вътрешн", r"\bbinnen")
LATERAL = rx(r"\blater(a|ij)l", r"\bexterno", r"\bexterne\b", r"\bdis\b", r"vanjsk", r"\bεξω\b", r"aussen",
             r"латерал", r"външн", r"\bbuiten")
PF = rx(r"patel", r"femoropatel", r"femoro-patel", r"retropatell", r"trochle", r"troclea", r"troklea", r"rotul",
        r"επιγονατιδ", r"μηριαιοεπιγονατιδ", r"τροχιλ", r"пател", r"капач", r"трохле", r"patell", r"\bfp\b", r"\bpf\b")
TF = rx(r"femorotibial", r"femoro-tibial", r"tibiofemoral", r"\bft\b", r"μηριαιοκνημια", r"феморотибиал", r"tibiofemoraal")

# ------------------------------------------------------------------ conditions
TEAR = rx(r"\btear", r"\btorn\b", r"ruptur", r"\brotur", r"\broto\b", r"desgarr", r"yirtik", r"kopma", r"puknuc",
          r"ρηξ", r"\briss", r"разкъс", r"руптур", r"scheur", r"dechirure", r"fissur", r"\blanguette", r"\bflap\b",
          r"bucket", r"\basa de cubo", r"maceration", r"macerac", r"fragment")
GRADE3 = rx(r"grad\w* (iii|3)\b", r"grade (iii|3)", r"(iii|3)\.? ?(°|stupn|grad)", r"\bevre (iii|3)")
SPRAIN = rx(r"sprain", r"esguince", r"distorz", r"distors", r"istegnuc", r"διαταση", r"zerrung", r"дисторз", r"verstuik",
            r"entorse", r"ozljed", r"κακωσ", r"\blesion", r"lezyon", r"\blezij", r"lasion", r"лезия", r"letsel",
            r"avuls", r"αποσπασ", r"ausriss", r"desinsertion", r"arrachement", r"discontinu", r"devamsiz", r"prekid",
            r"kontinuitatsunterbrech", r"\binjur", r"\bhasar")
OA = rx(r"osteoarthr", r"\barthros", r"gonarthr", r"artrosi", r"artroz", r"gonartr", r"osteoartr", r"αρθριτιδ",
        r"αρθροπαθ", r"\barthrose", r"артроз", r"гонартр", r"остеоартр", r"artrose", r"\boa\b",
        r"degenerative (joint|arthrop|change)", r"cambios degenerativos", r"dejeneratif (artrit|degisiklik)",
        r"degenerativn\w* promjen", r"εκφυλιστικ\w* (αλλοιωσ|οστεοαρθρ)", r"degenerative verander",
        r"дегенеративн\w* промен", r"degeneratieve")
CHONDRAL_HIGH = rx(r"(grade|grado|grad|stupn\w*|evre|βαθμ\w*|степен|graad|icrs|outerbridge)\s*(iv|iii|3|4)\b",
                   r"\b(iv|iii|3|4)\s*(°|\.?\s*stupn)", r"full[- ]thickness", r"espesor total", r"vollschicht", r"tam kat",
                   r"pune debljine", r"ολικ\w* (παχ|διαβρ|απωλ)", r"пълна дебелина", r"volledige dikte", r"pleine epaisseur",
                   r"denud", r"bone on bone", r"osteophyt", r"osteofit", r"οστεοφυτ", r"остеофит", r"osteophyten",
                   r"joint space narrow", r"pinzamiento", r"eklem aralig\w* daral", r"suzen\w* zglob", r"στενωσ\w* (του )?μεσαρθρ",
                   r"gelenkspalt\w* (verschmal|verengt)", r"стеснен", r"gewrichtsspleet\w* versmal", r"pincement")
CHONDRAL_ANY = rx(r"chondr", r"condr", r"kondr", r"hondr", r"χονδρ", r"хондр", r"cartilag", r"cartilago", r"kikirdak",
                  r"kartilaj", r"hrskavic", r"knorpel", r"хрущ", r"kraakbeen", r"cartilage")
EFFUSION = rx(r"effusion", r"hemarthros", r"haemarthros", r"lipohem", r"derrame", r"hemartros", r"efuzyon", r"efüzyon",
              r"sivi artis", r"\bizljev", r"hidrops", r"hydrops", r"συλλογ\w*", r"αιμαρθρ", r"erguss", r"hamarthros",
              r"излив", r"хемартроз", r"effusie", r"gewrichtsvocht", r"epanchement", r"hemarthrose", r"hydarthrose",
              r"intra-?articular fluid", r"joint fluid")
SYNOVITIS = rx(r"synovit", r"sinovit", r"συνοβιτ", r"синовит", r"synoviit", r"synovite",
               r"synovial (thicken|prolif|hypertroph)", r"(thicken\w*|hypertroph\w*) (of the )?synovi",
               r"engrosamiento sinovial", r"sinovyal (kalinlas|proliferas|hipertrof)", r"zadeblj\w* sinovij",
               r"παχυνσ\w* (του )?(αρθρικου υμενα|υμενα)", r"verdickt\w* synovi", r"synovialverdick", r"синовиалн\w* (задебел|удебел)",
               r"synoviale verdikking", r"epaississement synovial", r"pvns", r"villonodular")
HOFFA = rx(r"hoffa")
BAKER = rx(r"[bβб][aа]ker", r"бейкер", r"popliteal cyst", r"quiste popl", r"popliteal kist", r"poplitealn\w* cist",
           r"ιγνυακ\w* κυστ", r"poplitealzyste", r"подколенн\w* кист", r"popliteale cyste", r"kyste popl",
           r"gastrocnemi\w*[- ]semimembran\w* burs", r"semimembranosus[- ]gastrocnemius")
CONTUSION = rx(r"bone bruis", r"contusion", r"contusi", r"kontuzyon", r"kontuzij", r"kontusion", r"контузи", r"contusie",
               r"θλασ", r"bone marrow (o?edema|edema)", r"marrow (o?edema|edema)", r"edema oseo", r"edema de (la )?medula",
               r"kemik iligi odem", r"kemik iliginde odem", r"edem kosti", r"kostan\w* edem", r"kostanog edem", r"edem kostane srzi",
               r"οιδημα\w* (του )?(οστικου )?μυελ", r"οστικ\w* οιδημ", r"οστεομυελικ\w* οιδημ", r"knochenmark\w*odem",
               r"knochenodem", r"bone bruise", r"костномозъч\w* едем", r"костен\w* (оток|едем)", r"botoedeem",
               r"beenmergoedeem", r"oedeme osseux", r"oedeme medullaire", r"contusion osseuse")
FRACTURE = rx(r"fractur", r"kirik", r"fraktur", r"prijelom", r"καταγμ", r"фрактур", r"счупван", r"fractuur", r"\bbreuk",
              r"segond")

# ------------------------------------------------------------------ assertion cues
NEG_PRE = rx(r"\bno\b", r"\bnot\b", r"\bwithout\b", r"absence of", r"negative for", r"free of", r"\bnor\b",
             r"\bsin\b", r"\bni\b", r"ausencia", r"no se (observa|aprecia|identifica|evidencia|visualiza)", r"no hay",
             r"\bbez\b", r"\bnema\b", r"\bnije\b", r"ne (nalazi|prikazuje|vidi|uocava)", r"\bχωρις\b", r"\bδεν\b",
             r"\bουτε\b", r"απουσια", r"\bkein", r"\bohne\b", r"\bnicht\b", r"\bняма\b", r"\bбез\b", r"не се",
             r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"pas de", r"absence d", r"\bsans\b", r"\baucun", r"\bnon\b")
NEG_POST = rx(r"\byok", r"saptanmad", r"saptanmam", r"izlenmed", r"izlenmem", r"gorulmed", r"gorulmem", r"izlenmiyor",
              r"saptanmiyor", r"\bne vidi\b", r"\bnije\b", r"δεν (παρατηρ|αναγνωρ|ελεγχ|απεικον|διαπιστ)", r"\bне се\b",
              r"\bnicht\b", r"\bniet\b", r"\bnot seen\b", r"\bnot (identified|demonstrated|present)", r"\bexcluded\b",
              r"\bausgeschlossen\b", r"\buitgesloten\b", r"\bdescartad")
NORMAL = rx(r"\bintact", r"\bnormal", r"unremarkable", r"conservad", r"integr", r"preserv", r"\bolagan", r"korunmus",
            r"intakt", r"uredn", r"odrzan", r"ocuvan", r"ακεραι", r"φυσιολογικ", r"regelrecht", r"unauffallig", r"нормал",
            r"запазен", r"интакт", r"normaal", r"ongestoord", r"respect", r"sans anomalie", r"dentro de limites normales")
UNCERTAIN = rx(r"possibl", r"probabl", r"\blikely", r"suspect", r"suspicio", r"cannot be (excluded|ruled out)",
               r"may represent", r"suggest", r"equivocal", r"questionabl", r"\br/o\b", r"rule out", r"posible", r"sugier",
               r"sugest", r"sospech", r"no se puede descartar", r"dudos", r"olasi", r"muhtemel", r"supheli", r"ekarte",
               r"dislanam", r"dusundur", r"moguc", r"vjerojatn", r"sumnj", r"upitn", r"iskljuc", r"πιθαν", r"υποψ",
               r"υποπτ", r"ενδεχομ", r"αποκλει", r"fraglich", r"moglich", r"wahrscheinlich", r"verdacht", r"\bv\.? ?a\.",
               r"\bdd\b", r"auszuschlie", r"вероятн", r"възможн", r"съмнен", r"суспект", r"не се изключва", r"mogelijk",
               r"waarschijnlijk", r"verdenking", r"twijfel", r"uit te sluiten", r"douteu", r"evoqu", r"\bexclu", r"\?")
HISTORICAL = rx(r"status post", r"\bs/p\b", r"post-?op", r"postoperat", r"reconstruct", r"\bgraft", r"plast", r"menisc\w*ctom",
                r"menisektom", r"meniscektom", r"meniskektom", r"менискектом", r"μηνισκεκτομ", r"operad", r"operiert",
                r"opere\b", r"opereret", r"ameliyat", r"healed", r"consolidad", r"konsolid", r"old fracture", r"\bremote\b",
                r"cicatri", r"\bvroeger", r"\beski\b", r"stari(j|ja)\w* prijelom", r"παλαι", r"стара фрактура", r"ancien")
HISTORY_SENT = rx(r"\bhistory\b", r"clinical (information|indication|question|data|details)", r"\bindication",
                  r"antecedente", r"indicacion", r"motivo de", r"klinische (inlichtingen|gegevens|informatie)", r"vraagstel",
                  r"anamne", r"fragestellung", r"klinik:", r"klinische angaben", r"endikasyon", r"\bklinik\b", r"\boyku",
                  r"ιστορικ", r"κλινικ\w* (ερωτημα|πληροφορ|στοιχει)", r"анамнез", r"клинич\w* (данни|въпрос)",
                  r"renseignements", r"indication", r"klinicki podaci", r"uputna dijagnoza", r"razlog pregleda")
CLAUSE_SPLIT = re.compile(r"[,;]|\b(but|however|pero|aunque|ama|ancak|fakat|ali|međutim|αλλα|ομως|aber|jedoch|но|maar|mais)\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])(?<!\d\.)\s+|\n+|\s-\s|\s•\s|\s>\s")

STATE_SCORE = {"present": 0.9, "uncertain": 0.5, "historical": 0.3, "absent": 0.05, "not_mentioned": 0.15}
PRIORITY = {"present": 4, "uncertain": 3, "historical": 2, "absent": 1, "not_mentioned": 0}


SECTION_HDR = re.compile(
    r"(findings|hallazgos|bulgular|nalaz|ευρηματα|befund|мр находка|находк|bevindingen|resultats|"
    r"technique|tecnica|teknik|τεχνικ|техника|techniek|scanprotocol|impression|conclusi|sonuc|zakljuc|"
    r"συμπερασμ|beurteilung|заключение|conclusie|besluit|resultados|impresion|opinion|yorum)\s*:")


def sentences(text):
    # start a new sentence at every section header ("... seq bevindingen: ...")
    text = SECTION_HDR.sub(lambda m: "\n" + m.group(0), text)
    raw = [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    out, in_history = [], False
    for s in raw:
        is_hdr_hist = bool(HISTORY_SENT.search(s[:60]))
        if SECTION_HDR.match(s):
            in_history = False
            is_hdr_hist = False
        elif is_hdr_hist:
            in_history = True
        out.append((s, in_history or is_hdr_hist))
    return out


def assert_clause(clause, span):
    """Assertion of a concept found at span=(start,end) inside clause."""
    pre = clause[max(0, span[0] - 60):span[0]]
    post = clause[span[1]:span[1] + 50]
    if HISTORICAL.search(clause):
        return "historical"
    if NEG_PRE.search(pre) or NEG_POST.search(post):
        return "absent"
    if UNCERTAIN.search(clause):
        return "uncertain"
    return "present"


def find(pattern, clause):
    m = pattern.search(clause)
    return (m.start(), m.end()) if m else None


def extract(report):
    text = fold(report)
    res = {t: ("not_mentioned", "") for t in TARGETS}

    def upd(t, state, ev):
        if PRIORITY[state] > PRIORITY[res[t][0]]:
            res[t] = (state, ev[:180])

    for sent, is_hist in sentences(text):
        if is_hist:
            continue
        for clause in [c for c in CLAUSE_SPLIT.split(sent) if c and c.strip() and len(c) > 2]:
            # ligaments
            for tgt, anat in (("ACL", ACL), ("MCL", MCL)):
                a = find(anat, clause)
                if a:
                    c = find(TEAR, clause) or find(SPRAIN, clause)
                    if c:
                        upd(tgt, assert_clause(clause, c), clause)
                    elif NORMAL.search(clause) or NEG_PRE.search(clause):
                        upd(tgt, "absent", clause)
                    elif re.search(r"mucoid|mukoid|miksoid|myxoid|mucoide|βλεννοειδ|мукоид|degenerat|dejenera|εκφυλ|дегенерат", clause):
                        upd(tgt, "uncertain" if tgt == "ACL" else "absent", clause)
            # collective ligament statements ("cruciate ligaments intact")
            for tgt, coll in (("ACL", CRUCIATES), ("MCL", COLLATERALS)):
                if coll.search(clause) and not (find(TEAR, clause) or find(SPRAIN, clause)) and (NORMAL.search(clause) or NEG_PRE.search(clause)):
                    upd(tgt, "absent", clause)
            # menisci
            m = find(MENISC, clause)
            if m:
                med, lat = bool(MEDIAL.search(clause)), bool(LATERAL.search(clause))
                both = bool(re.search(r"both|ambos|her iki|oba|αμφοτερ|beide|двата|двете|deux|bilateral", clause)) or (not med and not lat)
                sides = (["Medial Meniscus"] if med or both else []) + (["Lateral Meniscus"] if lat or both else [])
                c = find(TEAR, clause) or find(GRADE3, clause)
                for tgt in sides:
                    if c:
                        st = assert_clause(clause, c)
                        if both and not (med or lat) and st == "present":
                            st = "uncertain"  # side not stated
                        upd(tgt, st, clause)
                    elif NORMAL.search(clause) or NEG_PRE.search(clause):
                        upd(tgt, "absent", clause)
            # osteoarthritis
            o = find(OA, clause) or (find(CHONDRAL_HIGH, clause) if CHONDRAL_ANY.search(clause) or re.search(r"osteophyt|osteofit|οστεοφυτ|остеофит", clause) else None)
            if o:
                st = assert_clause(clause, o)
                is_pf, is_tf = bool(PF.search(clause)), bool(TF.search(clause))
                med, lat = bool(MEDIAL.search(clause)), bool(LATERAL.search(clause))
                tri = bool(re.search(r"tricompart|tricompartiment|trikompart|τρι.?διαμερ|трикомпарт|pancompart|global|general", clause))
                if tri:
                    for tgt in ("Medial OA", "Lateral OA", "PF OA"):
                        upd(tgt, st, clause)
                else:
                    if is_pf and not (med or lat):
                        upd("PF OA", st, clause)
                    if med:
                        upd("Medial OA", st, clause)
                    if lat:
                        upd("Lateral OA", st, clause)
                    if is_tf and not (med or lat):
                        upd("Medial OA", "uncertain" if st == "present" else st, clause)
                        upd("Lateral OA", "uncertain" if st == "present" else st, clause)
                    if not (is_pf or is_tf or med or lat) and find(OA, clause):
                        for tgt in ("Medial OA", "Lateral OA", "PF OA"):
                            upd(tgt, "uncertain" if st == "present" else st, clause)
            # standalone findings
            for tgt, pat in (("Effusion", EFFUSION), ("Baker's", BAKER), ("Contusion", CONTUSION), ("Fracture", FRACTURE),
                             ("Synovitis", SYNOVITIS)):
                f = find(pat, clause)
                if f:
                    st = assert_clause(clause, f)
                    if tgt == "Effusion" and re.search(r"physiolog|fisiolog|fizyolog|fiziolog|φυσιολογ|physiolog|физиолог|fysiolog", clause):
                        st = "absent"
                    if tgt == "Contusion" and st == "present" and re.search(r"degenerat|dejenera|subchondr|subcondr|subkondr|εκφυλ|дегенерат|reactiv|reaktiv", clause) \
                            and not re.search(r"bruis|contus|kontuz|kontus|θλασ|контуз|contusie", clause):
                        st = "uncertain"
                    upd(tgt, st, clause)
            if HOFFA.search(clause) and re.search(r"edema|oedem|odem|ödem|οιδημ|едем|oedeem|inflam|impinge|sinovit|synovit", clause):
                upd("Synovitis", "uncertain" if not assert_clause(clause, find(HOFFA, clause)) == "absent" else "absent", clause)
    return res


def extract_scores(report):
    r = extract(report)
    return {t: STATE_SCORE[s] for t, (s, _) in r.items()}, r

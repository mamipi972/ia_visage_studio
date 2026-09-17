#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Preuve par mutation : chaque correctif est remis en defaut, et le test doit
alors echouer.

Usage : python3 outils/tests_mutation.py   (necessite numpy et opencv)

Un test qui passe ne prouve rien tant qu'on n'a pas verifie qu'il sait
echouer. Pour chaque mecanisme livre, ce banc remet le comportement fautif
dans une copie du code et verifie que le controle correspondant passe au
rouge. S'il reste vert, c'est qu'il ne teste pas ce qu'on croit.

Trois pieges que ce banc traite explicitement :

  - **une mutation perimee ne teste plus rien.** Si le code a change et que le
    texte a remplacer n'existe plus, la mutation est comptee en echec, jamais
    ignoree en silence.
  - **une mutation peut etre sans effet pour une raison exterieure.** Quand
    une mutation ne change rien, le banc le dit, et c'est le banc qu'il faut
    alors suspecter avant le hasard.
  - **deux mecanismes redondants se rattrapent l'un l'autre.** Les desactiver
    un par un donne deux tests verts, donc deux fausses preuves. La table
    ci-dessous contient donc aussi une mutation qui les retire ensemble.
"""

import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_GREFFON = os.path.join(RACINE, "ia_visage_studio.py")

import tests_worker
import tests_unitaires
import tests_integration


# ---------------------------------------------------------------------------
# Mutations du worker.
# ---------------------------------------------------------------------------
MUTATIONS_WORKER = [
    {
        "nom": "le decalage de la mise en lettre-boite est oublie",
        "pourquoi": "la boite revient dans l'espace image sans retirer le "
                    "remplissage : le flou atterrit a cote du visage, sans la "
                    "moindre erreur",
        "remplacements": [(
            "    x1 = (cx - bl / 2.0 - dx) / echelle\n"
            "    y1 = (cy - bh / 2.0 - dy) / echelle\n",
            "    x1 = (cx - bl / 2.0) / echelle\n"
            "    y1 = (cy - bh / 2.0) / echelle\n")],
        "tests": ["test_geometrie_et_contrat"],
        "rouges": ["cas 1 : chaque zone traitee tombe sur le visage designe "
                   "(tolerance 2 px)"],
    },
    {
        "nom": "l'echelle de la mise en lettre-boite est oubliee",
        "pourquoi": "les coordonnees restent dans l'espace 640x640 du modele",
        "remplacements": [(
            "    larg = bl / echelle\n    haut = bh / echelle\n",
            "    larg = bl\n    haut = bh\n")],
        "tests": ["test_geometrie_et_contrat"],
        "rouges": ["cas 1 : chaque zone traitee tombe sur le visage designe "
                   "(tolerance 2 px)"],
    },
    {
        "nom": "le pivot reste en BGR",
        "pourquoi": "cv2.imread rend du BGR ; sans conversion, le modele voit "
                    "les canaux permutes et rend un resultat plausible mais faux",
        "remplacements": [(
            "        pivot = cv2.cvtColor(pour_ia[:, :, :3], cv2.COLOR_BGR2RGB)",
            "        pivot = pour_ia[:, :, :3]")],
        "tests": ["test_geometrie_et_contrat"],
        "rouges": ["cas 1 : le worker reussit"],
    },
    {
        "nom": "la normalisation du modele est ignoree (simple division par 255)",
        "pourquoi": "l'entree reste dans [0, 1] au lieu de [-1, 1] : le vrai "
                    "modele accepterait et rendrait des visages mediocres",
        "remplacements": [(
            "    x = rgb_u8.astype(np.float32) / 255.0\n"
            "    x = (x - moyenne) / (ecart if ecart else 1.0)\n",
            "    x = rgb_u8.astype(np.float32) / 255.0\n")],
        "tests": ["test_chainage_et_liberation"],
        "rouges": ["cas 2 : le sourire est reellement applique",
                   "cas 2 : aucune etape n'a du degrader"],
    },
    {
        "nom": "une source 16 bits est traitee comme du 8 bits",
        "pourquoi": "la division par 255 d'un uint16 monte a 257 et sature "
                    "l'image en blanc, sans lever d'exception",
        "remplacements": [(
            "        pour_ia = (brut.astype(np.float32) / 257.0)"
            ".clip(0, 255).astype(np.uint8)",
            "        pour_ia = brut.clip(0, 255).astype(np.uint8)")],
        "tests": ["test_seize_bits"],
        "rouges": ["cas 3 : le worker reussit"],
    },
    {
        "nom": "la composition part du pivot 8 bits au lieu de la source",
        "pourquoi": "en 16 bits, les pixels hors zone traitee sont alors "
                    "requantifies : l'image change partout sans que cela se voie",
        "remplacements": [(
            "                else:\n"
            "                    origine = brut[:, :, :3].astype(np.float32)\n",
            "                else:\n"
            "                    origine = (base_avant_visages[:, :, ::-1]"
            ".astype(np.float32)\n"
            "                               * facteur)\n")],
        "tests": ["test_seize_bits"],
        "rouges": ["cas 3 : aucun pixel modifie hors des zones, en 16 bits"],
    },
    {
        "nom": "le fondu n'atteint pas zero sur le bord de la zone",
        "pourquoi": "une couture apparait au bord du rectangle, et la promesse "
                    "que seuls les visages changent cesse d'etre exacte",
        "remplacements": [(
            "    return np.clip(distance / plume, 0.0, 1.0).astype(np.float32)",
            "    return np.clip((distance + 1.0) / plume, 0.0, 1.0)"
            ".astype(np.float32)")],
        "tests": ["test_geometrie_et_contrat"],
        "rouges": ["cas 1 : le bord exterieur de chaque zone est exactement "
                   "inchange"],
    },
    {
        "nom": "la colorisation reprend la luminance du modele",
        "pourquoi": "l'image est alors eclaircie ou assombrie par le modele, "
                    "ce qu'aucune inspection rapide ne distingue d'une "
                    "colorisation reussie",
        "remplacements": [(
            "    lab_modele[:, :, 0] = lab_origine[:, :, 0]\n", "")],
        "tests": ["test_colorisation_conserve_la_luminance"],
        "rouges": ["cas 5 : la luminance reste celle de l'original"],
    },
    {
        "nom": "la table de preseance du worker est desactivee",
        "pourquoi": "un conflit qui arriverait malgre l'arbitrage du greffon "
                    "ferait charger un modele sur un visage deja floute",
        "remplacements": [(
            "    operations = list(operations)\n"
            "    if \"anonymiser\" not in operations:\n",
            "    operations = list(operations)\n"
            "    if True:\n")],
        "tests": ["test_preseance_cote_worker"],
        "rouges": ["cas 6 : le conflit est rattrape et signale",
                   "cas 6 : aucun modele de visage n'a ete charge"],
    },
    {
        "nom": "le plafond memoire n'est jamais pose",
        "pourquoi": "une protection qui ne protege pas est pire qu'une "
                    "protection absente ; elle se prouve, elle ne se suppose pas",
        "remplacements": [(
            "        resource.setrlimit(resource.RLIMIT_AS, (limite, dur))\n", "")],
        "tests": ["test_plafond_memoire"],
        "rouges": ["cas 11 : une allocation au-dela du plafond est refusee par "
                   "le systeme"],
        "posix_seulement": True,
    },
    {
        "nom": "une etape rend un tableau en virgule flottante",
        "pourquoi": "le format pivot est rompu ; numpy accepterait la "
                    "conversion en silence et le resultat serait tronque",
        "remplacements": [(
            "    return cv2.GaussianBlur(vignette, (noyau, noyau), 0)",
            "    return cv2.GaussianBlur(vignette, (noyau, noyau), 0)"
            ".astype(np.float32)")],
        "tests": ["test_geometrie_et_contrat"],
        "rouges": ["cas 1 : le worker reussit"],
    },
    {
        "nom": "une etape rend un tableau flottant ET le controle du pivot est "
               "neutralise",
        "pourquoi": "demonstration que c'est bien verifier_pivot qui rattrape "
                    "le cas precedent : sans lui, le worker produit une image "
                    "silencieusement tronquee et le test redevient vert",
        "remplacements": [
            ("    return cv2.GaussianBlur(vignette, (noyau, noyau), 0)",
             "    return cv2.GaussianBlur(vignette, (noyau, noyau), 0)"
             ".astype(np.float32)"),
            ("    if tableau is None:\n"
             "        raise ValueError(\"pivot absent en sortie de \" + ou)\n",
             "    return tableau\n"
             "    if tableau is None:\n"
             "        raise ValueError(\"pivot absent en sortie de \" + ou)\n")],
        "tests": ["test_geometrie_et_contrat"],
        "verts": ["cas 1 : le worker reussit"],
    },
]


# ---------------------------------------------------------------------------
# Mutations du greffon.
# ---------------------------------------------------------------------------
MUTATIONS_GREFFON = [
    {
        "nom": "la table de preseance du greffon est desactivee",
        "pourquoi": "le fichier de parametres partirait avec des intentions "
                    "contradictoires, et l'ordre d'application tiendrait du "
                    "hasard",
        "remplacements": [(
            "    retenues = [op for op in demandees if op in LIBELLES_OPERATIONS]\n"
            "    if \"anonymiser\" not in retenues:\n",
            "    retenues = [op for op in demandees if op in LIBELLES_OPERATIONS]\n"
            "    if True:\n")],
        "tests": ["test_arbitrage_des_intentions"],
        "rouges": ["anonymisation prioritaire : sourire et amelioration retires",
                   "le conflit laisse une trace explicite"],
    },
    {
        "nom": "le marqueur d'environnement redevient commun a la suite",
        "pourquoi": "deux greffons de versions differentes partageant la meme "
                    "pile se declareraient mutuellement perimes et "
                    "reinstalleraient plusieurs centaines de megaoctets a "
                    "chaque bascule, sans le moindre message",
        "remplacements": [(
            "    return \"env_\" + pile + \"_\" + PLUGIN_ID + \".json\"",
            "    return \"env_\" + pile + \".json\"")],
        "tests": ["test_emplacements_et_marqueur"],
        "rouges": ["le marqueur porte le nom du greffon"],
    },
    {
        "nom": "la reparation d'un environnement existant est supprimee",
        "pourquoi": "toute la logique de choix placee dans la seule fonction "
                    "de creation ne s'execute jamais sur une installation deja "
                    "en place : le greffon reinstallerait par-dessus le venv "
                    "d'un voisin de la suite, plusieurs centaines de "
                    "megaoctets, a chaque premier lancement",
        "remplacements": [(
            "    if not reinstaller and os.path.isfile(py):\n"
            '        progression("Verification de l\'environnement IA...")\n',
            "    if False and os.path.isfile(py):\n"
            '        progression("Verification de l\'environnement IA...")\n')],
        "tests": ["test_poste_deja_installe"],
        "rouges": ["aucune commande pip n'est lancee"],
    },
    {
        "nom": "la purge des journaux redevient alphabetique sur tout le dossier",
        "pourquoi": "logs/ est partage par la suite et deux conventions de "
                    "nommage y cohabitent ; en ASCII le tiret precede le "
                    "chiffre, donc un tri alphabetique supprimerait toujours "
                    "les archives du voisin avant les siennes",
        "remplacements": [(
            "        purger_journaux(racine)\n        return cible\n",
            "        entrees = sorted(os.path.join(racine, d)\n"
            "                         for d in os.listdir(racine)\n"
            "                         if os.path.isdir(os.path.join(racine, d)))\n"
            "        for vieux in entrees[:-ARCHIVES_A_CONSERVER]:\n"
            "            shutil.rmtree(vieux, ignore_errors=True)\n"
            "        return cible\n")],
        "tests": ["test_journaux_et_messages"],
        "rouges": ["les archives d'un greffon voisin sont intactes"],
    },
    {
        "nom": "les archives orphelines ne sont plus jamais resorbees",
        "pourquoi": "le stock laisse par les versions anterieures au marqueur "
                    "resterait indefiniment, et un greffon qui cree des "
                    "fichiers hors de son dossier temporaire doit savoir y "
                    "faire le menage",
        "remplacements": [(
            "    condamnees += [chemin for date, _, chemin\n"
            "                   in orphelines[:-ARCHIVES_A_CONSERVER] "
            "if date < limite]\n", "")],
        "tests": ["test_journaux_et_messages"],
        "rouges": ["les archives orphelines anciennes se resorbent"],
    },
    {
        "nom": "les donnees volumineuses retournent sous le profil GIMP",
        "pourquoi": "sur un profil itinerant, plusieurs gigaoctets seraient "
                    "synchronises a chaque ouverture de session",
        "remplacements": [(
            "    base = base_donnees()\n"
            "    cible = os.path.join(base, \"GIMP\", SHARED_DIR_NAME)\n",
            "    base = base_donnees()\n"
            "    cible = os.path.join(get_shared_dir(), \"donnees\")\n")],
        "tests": ["test_emplacements_et_marqueur"],
        "rouges": ["les donnees volumineuses sortent du profil GIMP"],
    },
    {
        "nom": "l'extraction du chemin de py -0p redevient un split sur les "
               "espaces",
        "pourquoi": "un chemin contenant Program Files est silencieusement "
                    "tronque, et la strategie du lanceur py meurt sans symptome",
        "remplacements": [(
            "MOTIF_LANCEUR_PY = r\"([a-zA-Z]:\\\\[^\\r\\n]*?python(?:[0-9._]*)"
            "\\.exe)\"",
            "MOTIF_LANCEUR_PY = r\"(\\S+python(?:[0-9._]*)\\.exe)\"")],
        "tests": ["test_detection_lanceur_py"],
        "rouges": ["un chemin contenant des espaces n'est pas tronque"],
    },
    {
        "nom": "le classement des interpreteurs ignore le canal de decouverte",
        "pourquoi": "un Python embarque dans une application tierce est alors "
                    "retenu, et disparait a la mise a jour de cette application "
                    "en emportant l'environnement",
        "remplacements": [(
            "    return sorted(candidats,\n"
            "                  key=lambda d: (d.get(\"score_canal\", 0), "
            "d.get(\"dans_plafond\", 0),\n"
            "                                 d.get(\"version\", (0,))),\n"
            "                  reverse=True)",
            "    return sorted(candidats,\n"
            "                  key=lambda d: d.get(\"version\", (0,)),\n"
            "                  reverse=True)")],
        "tests": ["test_classement_des_interpreteurs"],
        "rouges": ["le canal declare au systeme passe avant la version"],
    },
    {
        "nom": "le seuil de telechargement automatique est retire",
        "pourquoi": "plusieurs centaines de megaoctets partiraient sans que "
                    "l'utilisateur l'ait demande",
        "remplacements": [(
            "        if reference > AUTO_DOWNLOAD_MAX_BYTES:\n"
            "            echecs.append(\"%s : %s annonces, au-dela du seuil de %s\"\n",
            "        if False:\n"
            "            echecs.append(\"%s : %s annonces, au-dela du seuil de %s\"\n")],
        "tests": ["test_telechargement_annonce_et_seuil"],
        "rouges": ["aucun octet du corps n'est transfere lors d'un refus"],
    },
    {
        "nom": "l'avertissement materiel est conditionne par la detection",
        "pourquoi": "le message ne s'afficherait plus quand la detection echoue, "
                    "c'est-a-dire precisement quand l'utilisateur en a besoin",
        "remplacements": [(
            "    if not gpu_demande or materiel != \"processeur\":",
            "    if not sonde_runtime_cuda()[0] or materiel != \"processeur\":")],
        "tests": ["test_ecart_materiel_signale_une_fois"],
        "rouges": ["avec demande de GPU, l'ecart est signale"],
    },
    {
        "nom": "les processus enfants ne sont plus tues a la sortie",
        "pourquoi": "un worker survivrait a son parent et saturerait le "
                    "processeur en arriere-plan apres l'abandon",
        "remplacements": [(
            "            if proc.poll() is None:\n"
            "                tuer_groupe(proc)\n",
            "            if False:\n"
            "                tuer_groupe(proc)\n")],
        "tests": ["test_annulation_et_processus_orphelins"],
        "rouges": ["le filet de securite tue ce qui reste a la sortie"],
    },
    {
        "nom": "le filtrage du PATH devient une purge large",
        "pourquoi": "un runtime CUDA installe a l'echelle de la machine serait "
                    "coupe, et l'acceleration disparaitrait sans explication",
        "remplacements": [(
            "        if \"gimp\" in bas:\n            continue\n",
            "        if \"gimp\" in bas or \"cuda\" in bas:\n            continue\n")],
        "tests": ["test_isolation_environnement"],
        "rouges": ["le filtrage reste chirurgical : le runtime CUDA reste sur "
                   "le PATH"],
    },
]


# ---------------------------------------------------------------------------
# Mutations verifiees par le banc d'integration, qui exerce run_procedure du
# debut a la fin. Ce sont les seules qui portent sur l'orchestration : ce que
# le greffon fait du resultat du worker, et ce qu'il en dit a l'utilisateur.
# ---------------------------------------------------------------------------
MUTATIONS_INTEGRATION = [
    {
        "nom": "le calque produit est insere sous l'original",
        "pourquoi": "l'ordre d'empilement se verifie, il ne se deduit pas du "
                    "nom du parametre : position = -1 empile vers le haut, et "
                    "un decalage d'un rang cache le resultat sous le calque "
                    "source",
        "remplacements": [(
            "            image.insert_layer(calque, parent, position)\n",
            "            image.insert_layer(calque, parent, position + 1)\n")],
        "tests": ["test_anonymisation_de_bout_en_bout"],
        "rouges": ["un calque a ete ajoute au-dessus de l'original"],
    },
    {
        "nom": "le nom du calque ne nomme plus le moteur ni le materiel",
        "pourquoi": "le journal du worker disparait avec le dossier "
                    "temporaire : le nom du calque est la seule trace "
                    "persistante de ce qui a reellement servi",
        "remplacements": [(
            '        morceaux.append("%s %s sur %s" % (etape, moteur, materiel))',
            '        morceaux.append(etape)')],
        "tests": ["test_anonymisation_de_bout_en_bout"],
        "rouges": ["le nom du calque nomme le moteur reellement employe"],
    },
    {
        "nom": "les journaux ne sont plus archives",
        "pourquoi": "la destruction du dossier d'execution emporterait alors "
                    "la seule trace exploitable, et aucun signalement ne "
                    "serait utilisable",
        "remplacements": [(
            "def archiver_journaux(dossier_travail, contexte=None):\n    try:\n",
            "def archiver_journaux(dossier_travail, contexte=None):\n"
            "    return None\n    try:\n")],
        "tests": ["test_echec_archive_les_journaux"],
        "rouges": ["les journaux sont bien sur le disque",
                   "le message cite le dossier d'archivage"],
    },
    {
        "nom": "le dossier d'execution n'est plus detruit",
        "pourquoi": "chaque lancement laisserait une image et un script dans "
                    "le dossier temporaire global, indefiniment",
        "remplacements": [(
            "                shutil.rmtree(dossier_travail, ignore_errors=True)\n",
            "                pass\n")],
        "tests": ["test_dossier_temporaire_detruit"],
        "rouges": ["aucun dossier de travail ne reste dans le temp global"],
    },
    {
        "nom": "les degradations du worker ne remontent plus a l'utilisateur",
        "pourquoi": "une etape ignoree en silence est indiscernable d'une "
                    "etape reussie : l'utilisateur croirait avoir obtenu ce "
                    "qu'il a demande",
        "remplacements": [(
            '            for ligne in (resultat.get("degradations") or []):\n',
            '            for ligne in []:\n')],
        "tests": ["test_degradation_de_bout_en_bout"],
        # Deux mecanismes annoncent l'etape ignoree : la resolution des
        # modeles cote greffon, et la degradation constatee par le worker. Le
        # premier controle reste donc vert, et c'est normal ; seul le second
        # porte sur le chemin que cette mutation coupe.
        "rouges": ["la degradation constatee par le worker remonte telle quelle"],
    },
    {
        "nom": "un echec d'enregistrement n'est plus isole",
        "pourquoi": "une exception sur un seul appel ferait disparaitre le "
                    "greffon en entier, des menus comme du navigateur de "
                    "procedures, sans aucun message",
        "remplacements": [(
            "        for methode, arguments in enregistrements:\n"
            "            try:\n"
            "                getattr(proc, methode)(*arguments)\n"
            "            except Exception as e:\n"
            '                journal("enregistrement %s en echec: %s" % (methode, e))\n',
            "        for methode, arguments in enregistrements:\n"
            "            getattr(proc, methode)(*arguments)\n")],
        "tests": ["test_enregistrement_protege"],
        # La mutation ne rend pas un controle rouge : elle fait remonter
        # l'exception jusqu'a l'appelant, c'est-a-dire jusqu'a GIMP, qui fait
        # alors disparaitre le greffon en entier sans un mot. C'est
        # exactement le symptome qu'on veut prouver.
        "incident_attendu": True,
    },
]


# Les deux tables de preseance se rattrapent l'une l'autre. Les desactiver
# separement donne deux tests verts - deux fausses preuves - parce que chaque
# mecanisme couvre seul le cas signale. Ce n'est qu'en les retirant tous les
# deux que le defaut d'origine reapparait : des operations contradictoires
# atteignent les modeles. La redondance est une bonne conception et une
# mauvaise demonstration : on la garde, et on l'ecrit noir sur blanc.
MUTATION_CONJOINTE = {
    "nom": "les DEUX tables de preseance sont desactivees ensemble",
    "pourquoi": "demonstration que la redondance est voulue : chacune rattrape "
                "seule le cas signale, et aucun test ne discrimine autre chose "
                "que leur absence conjointe",
    "greffon": [(
        "    retenues = [op for op in demandees if op in LIBELLES_OPERATIONS]\n"
        "    if \"anonymiser\" not in retenues:\n",
        "    retenues = [op for op in demandees if op in LIBELLES_OPERATIONS]\n"
        "    if True:\n")],
    "worker": [(
        "    operations = list(operations)\n"
        "    if \"anonymiser\" not in operations:\n",
        "    operations = list(operations)\n"
        "    if True:\n")],
    "tests_greffon": ["test_arbitrage_des_intentions"],
    "tests_worker": ["test_preseance_cote_worker"],
    "rouges": ["anonymisation prioritaire : sourire et amelioration retires",
               "cas 6 : aucun modele de visage n'a ete charge"],
}


RESULTATS = []


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    suffixe = (" - " + str(detail)) if (detail and not condition) else ""
    print("  %s %s%s" % (marque, nom, suffixe))
    return bool(condition)


def appliquer(source, remplacements, nom_mutation):
    """Applique les remplacements, en refusant toute mutation perimee.

    Une mutation dont le texte n'existe plus ne teste rien, et le dire est la
    seule facon de ne pas s'en apercevoir trois versions plus tard.
    """
    for ancien, nouveau in remplacements:
        occurrences = source.count(ancien)
        if occurrences != 1:
            raise AssertionError(
                "mutation perimee (%s) : le motif apparait %d fois au lieu "
                "d'une.\nMotif :\n%s" % (nom_mutation, occurrences, ancien))
        source = source.replace(ancien, nouveau)
    return source


def executer(module, noms_de_tests):
    """Execute les tests demandes en silence, et rend leurs resultats."""
    module.RESULTATS[:] = []
    flux = io.StringIO()
    ancien_stdout = sys.stdout
    sys.stdout = flux
    try:
        for nom in noms_de_tests:
            getattr(module, nom)()
    except BaseException as e:
        sys.stdout = ancien_stdout
        return list(module.RESULTATS), "%s: %s" % (type(e).__name__, e)
    finally:
        sys.stdout = ancien_stdout
    return list(module.RESULTATS), ""


def verifier_attentes(mutation, resultats, incident):
    etats = dict((nom, ok) for nom, ok, _ in resultats)
    problemes = []
    for attendu in mutation.get("rouges", []):
        if attendu not in etats:
            problemes.append("le controle %r n'a pas ete execute" % attendu)
        elif etats[attendu]:
            problemes.append("le controle %r reste vert : il ne teste pas ce "
                             "qu'on croit" % attendu)
    for attendu in mutation.get("verts", []):
        if attendu not in etats:
            problemes.append("le controle %r n'a pas ete execute" % attendu)
        elif not etats[attendu]:
            problemes.append("le controle %r est rouge alors qu'il devait "
                             "rester vert" % attendu)
    if mutation.get("incident_attendu"):
        if not incident:
            problemes.append("le test s'est termine normalement alors que la "
                             "mutation devait le faire echouer bruyamment")
    elif incident and not mutation.get("rouges") and not mutation.get("verts"):
        problemes.append("incident pendant l'execution : " + incident)
    return problemes


def mutations_du_worker():
    print("Mutations du worker")
    original = tests_worker.extraire_worker()
    for mutation in MUTATIONS_WORKER:
        if mutation.get("posix_seulement") and os.name == "nt":
            controler("mutation ignoree sous Windows : " + mutation["nom"], True)
            continue
        try:
            mute = appliquer(original, mutation["remplacements"], mutation["nom"])
        except AssertionError as e:
            controler("mutation applicable : " + mutation["nom"], False, str(e))
            continue
        tests_worker.extraire_worker = lambda _source=mute: _source
        try:
            resultats, incident = executer(
                tests_worker, [mutation["tests"][0]] if len(mutation["tests"]) == 1
                else mutation["tests"])
        finally:
            tests_worker.extraire_worker = _ORIGINAL_EXTRAIRE_WORKER
        problemes = verifier_attentes(mutation, resultats, incident)
        controler("mutation detectee : " + mutation["nom"], not problemes,
                  " | ".join(problemes))


def mutations_du_greffon():
    print("Mutations du greffon")
    original = open(CHEMIN_GREFFON, encoding="utf-8").read()
    for mutation in MUTATIONS_GREFFON:
        try:
            mute = appliquer(original, mutation["remplacements"], mutation["nom"])
        except AssertionError as e:
            controler("mutation applicable : " + mutation["nom"], False, str(e))
            continue
        dossier = tempfile.mkdtemp(prefix="mutation_greffon_")
        chemin = os.path.join(dossier, "ia_visage_studio.py")
        with open(chemin, "w", encoding="utf-8", newline="\n") as f:
            f.write(mute)
        tests_unitaires.CHEMIN_GREFFON = chemin
        try:
            resultats, incident = executer(tests_unitaires, mutation["tests"])
        finally:
            tests_unitaires.CHEMIN_GREFFON = CHEMIN_GREFFON
            shutil.rmtree(dossier, ignore_errors=True)
        problemes = verifier_attentes(mutation, resultats, incident)
        controler("mutation detectee : " + mutation["nom"], not problemes,
                  " | ".join(problemes))


def mutations_d_integration():
    print("Mutations verifiees de bout en bout")
    original = open(CHEMIN_GREFFON, encoding="utf-8").read()
    for mutation in MUTATIONS_INTEGRATION:
        try:
            mute = appliquer(original, mutation["remplacements"], mutation["nom"])
        except AssertionError as e:
            controler("mutation applicable : " + mutation["nom"], False, str(e))
            continue
        dossier = tempfile.mkdtemp(prefix="mutation_integration_")
        chemin = os.path.join(dossier, "ia_visage_studio.py")
        with open(chemin, "w", encoding="utf-8", newline="\n") as f:
            f.write(mute)
        tests_integration.CHEMIN_GREFFON = chemin
        try:
            resultats, incident = executer(tests_integration, mutation["tests"])
        finally:
            tests_integration.CHEMIN_GREFFON = CHEMIN_GREFFON
            shutil.rmtree(dossier, ignore_errors=True)
        problemes = verifier_attentes(mutation, resultats, incident)
        controler("mutation detectee : " + mutation["nom"], not problemes,
                  " | ".join(problemes))


def mutation_conjointe():
    print("Mutation conjointe des deux mecanismes redondants")
    mutation = MUTATION_CONJOINTE
    source_greffon = open(CHEMIN_GREFFON, encoding="utf-8").read()
    source_worker = _ORIGINAL_EXTRAIRE_WORKER()
    try:
        greffon_mute = appliquer(source_greffon, mutation["greffon"],
                                 mutation["nom"])
        worker_mute = appliquer(source_worker, mutation["worker"], mutation["nom"])
    except AssertionError as e:
        controler("mutation applicable : " + mutation["nom"], False, str(e))
        return

    dossier = tempfile.mkdtemp(prefix="mutation_conjointe_")
    chemin = os.path.join(dossier, "ia_visage_studio.py")
    with open(chemin, "w", encoding="utf-8", newline="\n") as f:
        f.write(greffon_mute)
    tests_unitaires.CHEMIN_GREFFON = chemin
    tests_worker.extraire_worker = lambda _source=worker_mute: _source
    try:
        resultats_greffon, incident_g = executer(tests_unitaires,
                                                 mutation["tests_greffon"])
        resultats_worker, incident_w = executer(tests_worker,
                                                mutation["tests_worker"])
    finally:
        tests_unitaires.CHEMIN_GREFFON = CHEMIN_GREFFON
        tests_worker.extraire_worker = _ORIGINAL_EXTRAIRE_WORKER
        shutil.rmtree(dossier, ignore_errors=True)

    problemes = verifier_attentes(mutation, resultats_greffon + resultats_worker,
                                  incident_g or incident_w)
    controler("mutation detectee : " + mutation["nom"], not problemes,
              " | ".join(problemes))
    print("  NOTE  chacun des deux mecanismes rattrape seul le cas signale : "
          "le test ne discrimine que leur absence conjointe.")


_ORIGINAL_EXTRAIRE_WORKER = tests_worker.extraire_worker


def main():
    print("Preuve par mutation : chaque correctif remis en defaut doit faire "
          "echouer son test")
    print("")
    mutations_du_worker()
    print("")
    mutations_du_greffon()
    print("")
    mutations_d_integration()
    print("")
    mutation_conjointe()
    print("")
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d mutations, %d non detectee(s)" % (len(RESULTATS), len(echecs)))
    for nom in echecs:
        print("  ECHEC : " + nom)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())

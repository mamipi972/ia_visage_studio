#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Controles de livraison du greffon.

Usage : python3 outils/verifier_livraison.py

Verifie, sur le fichier livre :

  1. aucun caractere au-dela de 0x7F (un accent dans un commentaire suffit a
     faire disparaitre le greffon des menus sous Windows, sans aucun message) ;
  2. fins de ligne LF pures ;
  3. nom de fichier neutre, sans accent ni suffixe de version ;
  4. syntaxe du greffon, du worker et du script de validation qu'il embarque ;
  5. marqueurs de diagnostic (prefixes OK_, ERR_, INFO_) identiques des deux
     cotes : le worker etant ecrit sans interpolation, la liste est dupliquee
     et derive silencieusement ;
  6. coherence des cles de configuration entre le greffon et son worker :
     toute cle lue et jamais ecrite ferait travailler le worker avec une autre
     valeur que celle livree, sans rien signaler ;
  7. fidelite de TABLE_DES_VALEURS.md : chaque constante citee par la
     documentation est comparee a la valeur reellement definie dans le code ;
  8. coherence de suite : les ressources partagees avec les autres greffons
     portent les memes noms, et le marqueur d'environnement, lui, est bien
     propre a ce greffon ;
  9. somme de controle SHA-256, a publier a cote du fichier : c'est le seul
     moyen pour un utilisateur de distinguer un greffon defectueux d'un
     fichier altere pendant le transport.

Sortie : code 0 si tout passe, 1 sinon. Aucune dependance externe.
"""

import ast
import hashlib
import os
import re
import shutil
import sys
import tempfile
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FICHIER_GREFFON = os.path.join(RACINE, "ia_visage_studio.py")
TABLE_DES_VALEURS = os.path.join(RACINE, "TABLE_DES_VALEURS.md")

# Ressources partagees par toute la suite. Un nom modifie d'un seul cote
# separe silencieusement deux greffons qui devaient se partager un venv de
# plusieurs centaines de megaoctets, ou pire, les fait se le reinstaller
# mutuellement. Ces valeurs se relisent avec celles du greffon voisin avant
# toute livraison.
SUITE_ATTENDUE = {
    "SHARED_DIR_NAME": "ai_suite_shared",
    "VARIABLE_DOSSIER": "GIMP_AI_SUITE_DIR",
    "VARIABLE_DEBUG": "GIMP_AI_SUITE_DEBUG",
    "STACK_CPU": "onnx-cpu",
}
GREFFON_VOISIN = os.path.join(os.path.dirname(RACINE),
                              "gimp_sam2_segmentation",
                              "gimp_sam2_segmentation.py")

echecs = []
notes = []


def verifier(condition, message):
    if condition:
        notes.append("  OK    " + message)
    else:
        echecs.append(message)
        notes.append("  ECHEC " + message)


def extraire(source, nom):
    arbre = ast.parse(source)
    for noeud in arbre.body:
        if not isinstance(noeud, ast.Assign):
            continue
        for cible in noeud.targets:
            if getattr(cible, "id", "") == nom:
                try:
                    return ast.literal_eval(noeud.value)
                except Exception:
                    return None
    return None


def rendu_attendu(valeur):
    """Forme sous laquelle une constante doit apparaitre dans le tableau.

    Un tuple de nombres est un numero de version : (3, 8) s'ecrit 3.8. Un
    tuple ou une liste de chaines est une enumeration. Une liste de nombres -
    une normalisation, par exemple - s'enumere aussi : la traiter comme un
    numero de version donnerait 0.5.0.5, et le controle accuserait a tort une
    documentation pourtant juste.
    """
    if isinstance(valeur, tuple) and all(isinstance(v, (int, float))
                                         for v in valeur):
        return ".".join(str(v) for v in valeur)
    if isinstance(valeur, (tuple, list)):
        return ", ".join(str(v) for v in valeur)
    return str(valeur)


def charger_greffon_hors_gimp():
    """Charge le greffon avec la doublure, dans un bac isole.

    Retourne (module, bac, environnement_sauve) ou (None, None, None).
    """
    try:
        import faux_gimp
    except ImportError:
        return None, None, None
    bac = tempfile.mkdtemp(prefix="verif_livraison_")
    sauve = {cle: os.environ.get(cle) for cle in
             ("LOCALAPPDATA", "XDG_DATA_HOME", "HOME", "GIMP_AI_SUITE_DIR")}
    try:
        for cle in ("LOCALAPPDATA", "XDG_DATA_HOME", "HOME"):
            os.environ[cle] = bac
        os.environ.pop("GIMP_AI_SUITE_DIR", None)
        module = faux_gimp.installer(bac, FICHIER_GREFFON)
        return module, bac, sauve
    except Exception as e:
        verifier(False, "le greffon se charge hors de GIMP pour verification : %s" % e)
        shutil.rmtree(bac, ignore_errors=True)
        return None, None, None


def restaurer(bac, sauve):
    for cle, valeur in (sauve or {}).items():
        if valeur is None:
            os.environ.pop(cle, None)
        else:
            os.environ[cle] = valeur
    if bac:
        shutil.rmtree(bac, ignore_errors=True)


def verifier_table_des_valeurs(module):
    """Compare chaque constante citee par la documentation a celle du code.

    La verification de fidelite n'est praticable que si toutes les constantes
    citables sont regroupees dans un document unique, avec le nom de la
    constante en regard de sa valeur : c'est ce tableau qui rend l'ecart
    detectable.
    """
    if not os.path.isfile(TABLE_DES_VALEURS):
        verifier(False, "TABLE_DES_VALEURS.md est present")
        return
    if module is None:
        verifier(False, "la doublure faux_gimp.py est disponible")
        return

    texte = open(TABLE_DES_VALEURS, encoding="utf-8").read()
    citees = 0
    ecarts = []
    for ligne in texte.splitlines():
        if not ligne.startswith("|"):
            continue
        cellules = [c.strip() for c in ligne.strip("|").split("|")]
        if len(cellules) < 2:
            continue
        correspondance = re.match(r"^`([A-Z][A-Z0-9_]+)`$", cellules[0])
        if not correspondance:
            continue
        nom = correspondance.group(1)
        if not hasattr(module, nom):
            ecarts.append("%s : citee dans la documentation, absente du code" % nom)
            continue
        citees += 1
        attendu = rendu_attendu(getattr(module, nom))
        if attendu not in cellules[1]:
            ecarts.append("%s : code = %s, documentation = %s"
                          % (nom, attendu, cellules[1]))

    verifier(citees >= 30, "la table des valeurs cite les constantes du code "
                           "(%d citations)" % citees)
    verifier(not ecarts, "chaque valeur de la documentation correspond a la "
                         "constante du code (%s)" % ("; ".join(ecarts) or "aucun ecart"))


def verifier_coherence_de_suite(module, source):
    """Relecture croisee des noms de ressources partagees.

    Les conflits de ressources partagees sont invisibles a l'echelle d'un
    fichier : ils n'apparaissent qu'en regardant deux greffons ensemble.
    """
    if module is None:
        verifier(False, "coherence de suite verifiable (greffon chargeable)")
        return
    ecarts = []
    for nom, attendu in SUITE_ATTENDUE.items():
        obtenu = getattr(module, nom, None)
        if obtenu != attendu:
            ecarts.append("%s = %r, attendu %r" % (nom, obtenu, attendu))
    verifier(not ecarts,
             "les ressources partagees de la suite portent les noms attendus "
             "(%s)" % ("; ".join(ecarts) or "aucun ecart"))

    # Le marqueur d'environnement doit etre propre au greffon. S'il ne l'etait
    # pas, deux greffons de versions differentes partageant la meme pile se
    # declareraient mutuellement perimes et reinstalleraient plusieurs
    # centaines de megaoctets a chaque bascule, sans le moindre message.
    try:
        marqueur = module.nom_marqueur(module.STACK_CPU)
    except Exception as e:
        marqueur = ""
        ecarts.append(str(e))
    verifier(module.PLUGIN_ID in marqueur,
             "le marqueur d'environnement porte le nom du greffon (%s)" % marqueur)
    try:
        venv = module.nom_venv(module.STACK_CPU)
    except Exception:
        venv = ""
    verifier(venv == "venv-onnx-cpu" and module.PLUGIN_ID not in venv,
             "le venv est nomme par pile et partage avec la suite (%s)" % venv)
    verifier(module.nom_venv(module.STACK_CPU) != module.nom_venv(module.STACK_GPU),
             "les variantes processeur et GPU vivent dans des environnements "
             "distincts")

    if os.path.isfile(GREFFON_VOISIN):
        voisin = open(GREFFON_VOISIN, encoding="utf-8").read()
        desaccords = []
        for nom in ("SHARED_DIR_NAME", "VARIABLE_DOSSIER", "VARIABLE_DEBUG"):
            valeur_voisine = extraire(voisin, nom)
            if valeur_voisine is not None and valeur_voisine != getattr(module, nom):
                desaccords.append("%s : ici %r, voisin %r"
                                  % (nom, getattr(module, nom), valeur_voisine))
        paquets_voisins = extraire(voisin, "REQUIRED_PACKAGES") or []
        if paquets_voisins and sorted(paquets_voisins) != sorted(
                module.REQUIRED_PACKAGES):
            desaccords.append(
                "la pile processeur differe du greffon voisin alors que les "
                "deux partagent le meme venv : ici %s, voisin %s"
                % (sorted(module.REQUIRED_PACKAGES), sorted(paquets_voisins)))
        verifier(not desaccords,
                 "accord avec le greffon voisin sur les ressources communes "
                 "(%s)" % ("; ".join(desaccords) or "aucun desaccord"))
    else:
        notes.append("  NOTE  greffon voisin absent du poste : la relecture "
                     "croisee n'a porte que sur la table attendue")


def main():
    if not os.path.isfile(FICHIER_GREFFON):
        print("Fichier introuvable : " + FICHIER_GREFFON)
        return 1

    with open(FICHIER_GREFFON, "rb") as f:
        brut = f.read()
    source = brut.decode("utf-8")

    # 1. ASCII pur.
    hors_ascii = sorted({c for c in source if ord(c) > 127})
    if hors_ascii:
        apercu = ", ".join("%r (%s)" % (c, unicodedata.name(c, "?"))
                           for c in hors_ascii[:8])
        verifier(False, "caracteres au-dela de 0x7F dans le fichier livre : " + apercu)
    else:
        verifier(True, "aucun caractere au-dela de 0x7F")

    # 2. Fins de ligne.
    verifier(b"\r\n" not in brut and b"\r" not in brut, "fins de ligne LF pures")

    # 3. Nom du fichier livre.
    #
    # GIMP exige que le fichier porte le nom de son dossier d'installation. Le
    # depot, lui, garde le fichier a la racine : c'est le dossier
    # plug-ins/<nom>/ cree a l'installation qui doit correspondre, ce qu'aucun
    # controle ici ne peut verifier. On controle donc ce qui est verifiable :
    # un nom neutre, utilisable tel quel comme nom de dossier.
    nom_fichier = os.path.basename(FICHIER_GREFFON)
    verifier(re.match(r"^[a-z0-9_]+\.py$", nom_fichier) is not None,
             "nom de fichier neutre, sans accent, espace ni suffixe de version")
    notes.append("  RAPPEL a l'installation, le fichier va dans "
                 "plug-ins/%s/%s" % (nom_fichier[:-3], nom_fichier))

    # 4. Syntaxe du greffon et des scripts qu'il embarque.
    try:
        ast.parse(source)
        verifier(True, "syntaxe du greffon")
    except SyntaxError as e:
        verifier(False, "syntaxe du greffon : " + str(e))
        return 1

    worker = extraire(source, "WORKER_SCRIPT")
    verifier(isinstance(worker, str) and len(worker) > 100,
             "le script worker est present et non interpole")
    if isinstance(worker, str):
        try:
            ast.parse(worker)
            verifier(True, "syntaxe du script worker")
        except SyntaxError as e:
            verifier(False, "syntaxe du script worker : " + str(e))

    validation = extraire(source, "VALIDATION_SCRIPT")
    verifier(isinstance(validation, str) and len(validation) > 100,
             "le script de validation materielle est present et non interpole")
    if isinstance(validation, str):
        try:
            ast.parse(validation)
            verifier(True, "syntaxe du script de validation materielle")
        except SyntaxError as e:
            verifier(False, "syntaxe du script de validation materielle : " + str(e))

    # 5. Coherence des marqueurs.
    # Convention : un marqueur commence par OK_, ERR_ ou INFO_, ce qui le
    # distingue d'une indexation Python du type MODELES[ROLE_DETECTION].
    motif = r"\[(?:OK|ERR|INFO)_[A-Z_]+\]"
    declares = set(extraire(source, "MARQUEURS_WORKER") or ())
    dans_worker = set(re.findall(motif, worker or ""))
    hors_worker = set(re.findall(motif, source.replace(worker or "", "")))
    verifier(bool(declares), "la liste MARQUEURS_WORKER existe")
    verifier(declares == dans_worker,
             "marqueurs declares et marqueurs du worker identiques "
             "(manquants: %s ; en trop: %s)"
             % (sorted(declares - dans_worker) or "aucun",
                sorted(dans_worker - declares) or "aucun"))
    inconnus = hors_worker - declares
    verifier(not inconnus,
             "aucun marqueur inconnu cote greffon (%s)" % (sorted(inconnus) or "aucun"))

    # 6. Cles de configuration : tout ce que le worker lit doit etre ecrit.
    lues = set(re.findall(r'cfg\.get\(\s*"([a-z_]+)"', worker or ""))
    lues |= set(re.findall(r'cfg\[\s*"([a-z_]+)"\s*\]', worker or ""))
    hors = source.replace(worker or "", "")
    ecrites = set(re.findall(r'^\s+"([a-z_]+)":', hors, re.MULTILINE))
    oubliees = lues - ecrites
    verifier(not oubliees,
             "chaque parametre lu par le worker est ecrit par le greffon "
             "(sinon il retombe en silence sur une valeur par defaut) : %s"
             % (sorted(oubliees) or "aucun oubli"))
    notes.append("  NOTE  %d cle(s) de configuration lue(s) par le worker"
                 % len(lues))

    module, bac, sauve = charger_greffon_hors_gimp()
    try:
        # 7. Fidelite de la documentation.
        verifier_table_des_valeurs(module)
        # 8. Coherence de suite.
        verifier_coherence_de_suite(module, source)
    finally:
        restaurer(bac, sauve)

    # 9. Somme de controle.
    empreinte = hashlib.sha256(brut).hexdigest()

    print("Controles de livraison : " + FICHIER_GREFFON)
    for ligne in notes:
        print(ligne)
    print("")
    print("  Taille   : %d octets" % len(brut))
    print("  SHA-256  : " + empreinte)
    print("")
    if echecs:
        print("%d controle(s) en echec." % len(echecs))
        return 1
    print("Tous les controles passent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

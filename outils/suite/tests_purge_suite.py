#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Controles du correctif de purge des journaux, sur les quatre greffons.

Usage : python3 tests_purge_suite.py [dossier des greffons]

Sans argument, les greffons sont cherches dans le dossier parent de celui-ci.
Aucune dependance externe.

Le dossier logs/ est partage par toute la suite. Avant ce correctif, chaque
greffon y purgeait "tout sauf les N plus recents" en se fiant a l'ordre
alphabetique des noms de dossiers - et comme deux conventions de nommage
cohabitent, la purge d'un greffon emportait systematiquement les archives des
autres.

Ces controles portent sur des artefacts : ce qui reste sur le disque apres la
purge, et ce que contient l'archive produite. Jamais sur l'absence d'erreur.

Chaque greffon est reellement charge, avec une doublure de l'API GIMP : ce sont
ses propres fonctions qui sont exercees, pas une reimplementation.
"""

import json
import os
import shutil
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
# Les quatre greffons vivent hors de tout depot : le dossier qui les contient
# se passe donc en argument, et vaut par defaut le dossier parent de celui-ci.
RACINE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.dirname(ICI)
sys.path.insert(0, ICI)

import doublure_gimp

JOUR = 86400

GREFFONS = [
    {"fichier": "deep_erase.py", "identite": "PLUGIN_ID",
     "conserves": "KEPT_INCIDENTS", "dossier_logs": "get_logs_directory",
     "purge": "purge_log_archives", "archive": "archive_logs",
     "argument": "liste"},
    {"fichier": "deep_erase_pro.py", "identite": "PLUGIN_ID",
     "conserves": "KEPT_INCIDENTS", "dossier_logs": "get_logs_directory",
     "purge": "purge_log_archives", "archive": "archive_logs",
     "argument": "dossier"},
    {"fichier": "gimp_sam2_segmentation.py", "identite": "PLUGIN_ID",
     "conserves": "ARCHIVES_A_CONSERVER", "dossier_logs": "get_logs_dir",
     "purge": "purger_journaux", "archive": "archiver_journaux",
     "argument": "dossier"},
    {"fichier": "ia_detourage.py", "identite": "NOM_GREFFON",
     "conserves": "INCIDENTS_CONSERVES", "dossier_logs": "dossier_journaux",
     "purge": "_purger_journaux", "archive": "archiver_journaux",
     "argument": "dossier"},
]

RESULTATS = []


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    suffixe = (" - " + str(detail)) if (detail and not condition) else ""
    print("  %s %s%s" % (marque, nom, suffixe))
    return bool(condition)


class Bac(object):
    """Faux profil GIMP, et le greffon charge dedans."""

    def __init__(self, descripteur):
        self.descripteur = descripteur
        self.dossier = tempfile.mkdtemp(prefix="purge_suite_")
        self.profil = os.path.join(self.dossier, "profil_gimp")
        os.makedirs(self.profil, exist_ok=True)
        self.sauve = {}
        for cle in ("LOCALAPPDATA", "XDG_DATA_HOME", "HOME",
                    "GIMP_AI_SUITE_DIR"):
            self.sauve[cle] = os.environ.get(cle)
        for cle in ("LOCALAPPDATA", "XDG_DATA_HOME", "HOME"):
            os.environ[cle] = os.path.join(self.dossier, "donnees")
        os.environ.pop("GIMP_AI_SUITE_DIR", None)
        os.makedirs(os.path.join(self.dossier, "donnees"), exist_ok=True)
        self.module = doublure_gimp.installer(
            self.profil, os.path.join(RACINE, descripteur["fichier"]),
            "greffon_" + descripteur["fichier"][:-3])
        self.identite = getattr(self.module, descripteur["identite"])
        self.conserves = getattr(self.module, descripteur["conserves"])
        self.logs = getattr(self.module, descripteur["dossier_logs"])()
        self.purger = getattr(self.module, descripteur["purge"])
        self.archiver = getattr(self.module, descripteur["archive"])

    def travail(self):
        """Dossier d'execution factice, accepte par les quatre greffons."""
        chemin = tempfile.mkdtemp(prefix="travail_", dir=self.dossier)
        for nom in ("worker.log", "parametres.json"):
            with open(os.path.join(chemin, nom), "w", encoding="utf-8") as f:
                f.write("contenu de test\n")
        return chemin

    def archiver_une_fois(self):
        chemin = self.travail()
        if self.descripteur["argument"] == "liste":
            arg = [os.path.join(chemin, n)
                   for n in sorted(os.listdir(chemin))]
        else:
            arg = chemin
        cible = self.archiver(arg)
        shutil.rmtree(chemin, ignore_errors=True)
        return cible

    def poser_archive(self, nom, greffon=None, age_jours=0):
        """Depose une archive dans logs/, eventuellement au nom d'un voisin."""
        chemin = os.path.join(self.logs, nom)
        os.makedirs(chemin, exist_ok=True)
        with open(os.path.join(chemin, "worker.log"), "w",
                  encoding="utf-8") as f:
            f.write("x\n")
        if greffon is not None:
            marque = os.path.join(chemin, "incident.json")
            with open(marque, "w", encoding="utf-8") as f:
                json.dump({"greffon": greffon, "version": "1.0"}, f)
        if age_jours:
            quand = time.time() - age_jours * JOUR
            for racine, _, fichiers in os.walk(chemin):
                for f in fichiers:
                    os.utime(os.path.join(racine, f), (quand, quand))
            os.utime(chemin, (quand, quand))
        return chemin

    def inventaire(self):
        """(les miennes, celles des voisins, les non identifiees)."""
        miennes, voisines, muettes = [], [], []
        for nom in sorted(os.listdir(self.logs)):
            chemin = os.path.join(self.logs, nom)
            if not os.path.isdir(chemin):
                continue
            marque = os.path.join(chemin, "incident.json")
            if not os.path.isfile(marque):
                muettes.append(chemin)
                continue
            try:
                with open(marque, encoding="utf-8") as f:
                    greffon = json.load(f).get("greffon")
            except Exception:
                greffon = None
            if greffon == self.identite:
                miennes.append(chemin)
            else:
                voisines.append(chemin)
        return miennes, voisines, muettes

    def fermer(self):
        for cle, valeur in self.sauve.items():
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur
        shutil.rmtree(self.dossier, ignore_errors=True)


def purge_alphabetique(bac):
    """Le comportement fautif, remis en place pour la preuve par mutation."""

    def purge(base=None):
        base = base or bac.logs
        entrees = sorted(os.path.join(base, d) for d in os.listdir(base)
                         if os.path.isdir(os.path.join(base, d)))
        condamnees = entrees[:-bac.conserves]
        for chemin in condamnees:
            shutil.rmtree(chemin, ignore_errors=True)
        return condamnees

    return purge


def scenario_voisins(bac, prefixe):
    """Le cas reel : des archives a moi, celles d'un voisin, une non marquee.

    Les archives voisines portent l'autre convention de nommage de la suite.
    En ASCII le tiret precede le chiffre : elles se trient donc toujours en
    tete, et c'est exactement ce qui les faisait supprimer en premier.
    """
    voisines = [bac.poser_archive("2026-09-16_19-44-0%d" % i,
                                  greffon="un_autre_greffon")
                for i in (5, 6, 7)]
    muette = bac.poser_archive("2026-09-16_19-50-00")
    for _ in range(bac.conserves + 4):
        bac.archiver_une_fois()

    miennes, restantes_voisines, muettes = bac.inventaire()
    controler("%s : la purge conserve les %d archives de ce greffon"
              % (prefixe, bac.conserves),
              len(miennes) == bac.conserves, "%d restantes" % len(miennes))
    controler("%s : les archives du greffon voisin sont intactes" % prefixe,
              all(os.path.isdir(d) for d in voisines),
              str([d for d in voisines if not os.path.isdir(d)]))
    controler("%s : une archive recente non identifiee est conservee" % prefixe,
              os.path.isdir(muette), muette)
    return voisines, muette


def tester_greffon(descripteur):
    nom_court = descripteur["fichier"][:-3]
    print("Greffon : " + nom_court)
    bac = Bac(descripteur)
    try:
        # 1. L'archive produite porte bien le marqueur.
        cible = bac.archiver_une_fois()
        controler("%s : une archive est produite" % nom_court,
                  bool(cible) and os.path.isdir(cible), str(cible))
        if not cible or not os.path.isdir(cible):
            return
        marque = os.path.join(cible, "incident.json")
        controler("%s : l'archive porte un incident.json" % nom_court,
                  os.path.isfile(marque), marque)
        donnees = {}
        if os.path.isfile(marque):
            with open(marque, encoding="utf-8") as f:
                donnees = json.load(f)
        controler("%s : le marqueur nomme ce greffon et sa version" % nom_court,
                  donnees.get("greffon") == bac.identite
                  and bool(donnees.get("version")), str(donnees))
        controler("%s : les journaux sont bien copies a cote" % nom_court,
                  os.path.isfile(os.path.join(cible, "worker.log")),
                  str(sorted(os.listdir(cible))))

        # Deux incidents dans la meme seconde. C'est le cas qui compte : une
        # serie d'echecs rapproches est precisement celle ou les journaux
        # servent, et un horodatage a la seconde ferait que le second archivage
        # ecraserait le premier.
        seconde = bac.archiver_une_fois()
        controler("%s : deux incidents rapproches ne s'ecrasent pas" % nom_court,
                  bool(seconde) and os.path.normpath(seconde)
                  != os.path.normpath(cible),
                  "%s puis %s" % (cible, seconde))

        # 2. Le cas reel : purge en presence d'un voisin.
        scenario_voisins(bac, nom_court)
    finally:
        bac.fermer()

    # 3. Les orphelines anciennes se resorbent, les recentes non.
    bac = Bac(descripteur)
    try:
        anciennes = [bac.poser_archive("2025-01-0%d_00-00-00" % (i + 1),
                                       age_jours=40)
                     for i in range(bac.conserves + 3)]
        bac.purger()
        restantes = [d for d in anciennes if os.path.isdir(d)]
        controler("%s : les archives orphelines anciennes se resorbent"
                  % nom_court, len(restantes) == bac.conserves,
                  "%d restantes sur %d" % (len(restantes), len(anciennes)))
    finally:
        bac.fermer()

    bac = Bac(descripteur)
    try:
        peu = [bac.poser_archive("2025-02-0%d_00-00-00" % (i + 1),
                                 age_jours=40)
               for i in range(3)]
        bac.purger()
        controler("%s : une orpheline ancienne mais peu nombreuse est gardee"
                  % nom_court, all(os.path.isdir(d) for d in peu),
                  str([d for d in peu if not os.path.isdir(d)]))
    finally:
        bac.fermer()


def tester_mutation(descripteur):
    """Preuve par mutation : le comportement fautif doit faire echouer le test.

    Un controle qui passe ne prouve rien tant qu'on n'a pas verifie qu'il sait
    echouer. On remet donc la purge alphabetique sur tout le dossier, et on
    exige que le controle des archives voisines passe au rouge.
    """
    nom_court = descripteur["fichier"][:-3]
    bac = Bac(descripteur)
    avant = len(RESULTATS)
    try:
        setattr(bac.module, descripteur["purge"], purge_alphabetique(bac))
        scenario_voisins(bac, nom_court + " [mute]")
    finally:
        bac.fermer()
    produits = RESULTATS[avant:]
    del RESULTATS[avant:]
    cible = "%s [mute] : les archives du greffon voisin sont intactes" % nom_court
    etats = dict((n, ok) for n, ok, _ in produits)
    if cible not in etats:
        return controler("%s : mutation exercee" % nom_court, False,
                         "le controle vise n'a pas ete execute")
    return controler(
        "%s : purge alphabetique remise en place -> le controle passe au rouge"
        % nom_court, not etats[cible],
        "le controle reste vert : il ne teste pas ce qu'on croit")


def main():
    print("Correctif de purge des journaux - controles sur les quatre greffons")
    print("Greffons cherches dans : " + RACINE)
    print("")
    for descripteur in GREFFONS:
        if not os.path.isfile(os.path.join(RACINE, descripteur["fichier"])):
            controler("%s est present" % descripteur["fichier"], False)
            continue
        tester_greffon(descripteur)
    print("")
    print("Preuve par mutation")
    for descripteur in GREFFONS:
        if os.path.isfile(os.path.join(RACINE, descripteur["fichier"])):
            tester_mutation(descripteur)
    print("")
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d controles, %d echec(s)" % (len(RESULTATS), len(echecs)))
    for nom in echecs:
        print("  ECHEC : " + nom)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())

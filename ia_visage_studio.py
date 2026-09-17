#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Greffon GIMP 3.0 - IA Visage Studio : anonymiser, faire sourire, ameliorer et
coloriser, en une seule passe.

Quatre operations cumulatives, chainees dans un unique sous-processus :

  1. colorisation de l'image entiere (modele ONNX, chrominance seule) ;
  2. detection des visages (YOLOv8-face ONNX, repli cascade de Haar, repli
     selection) ;
  3. anonymisation des visages (flou ou pixellisation, sans aucune IA) ;
  4. sourire (AttGAN ONNX) puis amelioration (GFPGAN ONNX) sur les vignettes.

Architecture v6, augmentee de la section 25 du document de conception : une
table de preseance expurge les intentions contradictoires avant l'ecriture du
fichier de parametres, un format pivot unique circule entre les etapes, et
chaque modele est decharge avant le chargement du suivant.

Contrainte de livraison : ce fichier ne contient aucun caractere au-dela de
0x7F, commentaires et libelles d'interface compris. Les textes francais sont
donc ecrits sans accent. Voir outils/verifier_livraison.py.
"""

import os
import sys
import time
import json
import shutil
import hashlib
import tempfile
import subprocess

# Constantes necessaires avant le chargement de l'API, pour pouvoir ecrire un
# journal meme si ce chargement echoue.
PLUGIN_ID = "ia_visage_studio"
SHARED_DIR_NAME = "ai_suite_shared"
VARIABLE_DOSSIER = "GIMP_AI_SUITE_DIR"

# Versions de l'API GObject de GIMP essayees, dans l'ordre. "3.0" est l'API de
# GIMP 3.0, 3.2, 3.4... : le numero suit l'API, pas l'application. Une future
# GIMP 4 apporterait une API "4.0", que le greffon tente alors plutot que de
# disparaitre des menus sans un mot.
API_GIMP_CANDIDATES = ("3.0", "4.0")


def base_donnees():
    """Racine des donnees volumineuses, suivant les conventions du systeme."""
    if os.name == "nt":
        return os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support")
    return os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))


def journal_amorcage(message):
    """Journal ecrit sans l'API GIMP, pour les pannes qui la precedent.

    Si le chargement de l'API echoue, le greffon disparait des menus sans
    aucun message : ce fichier est alors la seule explication possible.
    """
    try:
        dossier = os.environ.get(VARIABLE_DOSSIER, "").strip() or os.path.join(
            base_donnees(), "GIMP", SHARED_DIR_NAME)
        os.makedirs(dossier, exist_ok=True)
        with open(os.path.join(dossier, "journal_amorcage.log"), "a",
                  encoding="utf-8") as f:
            f.write("%s %s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                    PLUGIN_ID, message))
    except Exception:
        pass


import gi

API_GIMP = None
_ERREURS_API = []
for _version in API_GIMP_CANDIDATES:
    try:
        gi.require_version("Gimp", _version)
        gi.require_version("GimpUi", _version)
        API_GIMP = _version
        break
    except Exception as _erreur:
        _ERREURS_API.append("%s: %s" % (_version, _erreur))

if API_GIMP is None:
    journal_amorcage(
        "aucune version connue de l'API GIMP n'est disponible (%s). Le greffon "
        "ne peut pas s'enregistrer ; il lui faut une mise a jour pour cette "
        "version de GIMP." % " | ".join(_ERREURS_API))
    raise ImportError("API GIMP indisponible : " + " | ".join(_ERREURS_API))

if API_GIMP != API_GIMP_CANDIDATES[0]:
    journal_amorcage(
        "API GIMP %s utilisee a la place de %s : comportement non teste."
        % (API_GIMP, API_GIMP_CANDIDATES[0]))

from gi.repository import Gimp
from gi.repository import GimpUi
from gi.repository import Gio
from gi.repository import GObject
from gi.repository import GLib

if os.name == "nt":
    import winreg

# ==============================================================================
# 1. IDENTITE ET CONSTANTES NOMMEES
#    Toute valeur citee dans la documentation vient d'ici (voir
#    TABLE_DES_VALEURS.md), et outils/verifier_livraison.py compare les deux.
# ==============================================================================
PLUGIN_VERSION = "1.1"
PROCEDURE_NAME = "plug-in-ia-visage-studio"

# --- Piles techniques ---------------------------------------------------------
# Le nom de la pile suffixe le venv, le marqueur et le cache d'interpreteur.
# onnxruntime et onnxruntime-gpu s'installent dans le meme dossier du
# site-packages et se detruisent mutuellement : un environnement par variante
# est la seule structure saine, et basculer de l'une a l'autre ne consiste
# jamais a installer par-dessus.
STACK_CPU = "onnx-cpu"
STACK_GPU = "onnx-gpu"

# Dependances, derivees des import du worker et de rien d'autre. Le worker
# importe numpy, cv2 et onnxruntime : il n'y a donc ni pillow, ni le paquet
# contrib d'OpenCV, ni le moindre extra de confort.
#
# La borne haute d'OpenCV n'est pas decorative : a partir d'OpenCV 5, le paquet
# ne livre plus les fichiers XML des cascades de Haar, dont depend le detecteur
# de repli. Le worker verifie malgre tout leur presence a l'execution plutot
# que de la deduire de cette borne.
PAQUETS_COMMUNS = [
    "numpy>=1.24.0,<3",
    "opencv-python-headless>=4.8.0,<5",
]
REQUIRED_PACKAGES = PAQUETS_COMMUNS + ["onnxruntime>=1.16.0,<2"]
REQUIRED_PACKAGES_GPU = PAQUETS_COMMUNS + ["onnxruntime-gpu>=1.16.0,<2"]

# Paquets pip de cuDNN, tentes dans cet ordre quand la pile GPU est demandee.
# Une dependance systeme que l'utilisateur ne peut pas installer sans quitter
# le greffon existe souvent sous forme de roue publiee par le meme editeur :
# la tenter ne coute rien, et son echec n'est jamais bloquant.
PAQUETS_CUDNN = ["nvidia-cudnn-cu12", "nvidia-cudnn-cu11"]

# Bornes de version de l'interpreteur. Le plancher est impose par les roues des
# dependances. Le plafond n'interdit rien, il exprime une preference : sur un
# poste qui ne possede qu'une version plus recente, cette version est retenue.
# A relever apres chaque campagne de test.
PY_MIN = (3, 8)
PY_MAX_TESTED = (3, 12)

# --- Modeles ------------------------------------------------------------------
# Plafond de vraisemblance d'un fichier de modele (section 9 du scenario) :
# au-dela, le fichier depose n'est manifestement pas un modele ONNX de cette
# suite. Sans aucun rapport avec AUTO_DOWNLOAD_MAX_BYTES ci-dessous.
MODELE_TAILLE_MIN_BYTES = 1 * 1024 * 1024
MODELE_TAILLE_MAX_BYTES = 2 * 1024 * 1024 * 1024

# Seuil de telechargement automatique, par fichier (section 10 du scenario).
# En deca, le telechargement est automatique mais jamais silencieux : la taille
# reelle annoncee par le serveur est affichee avant qu'il ne demarre. Au-dela,
# le greffon refuse, affiche le chemin exact de depot, et l'operation concernee
# degrade vers son repli sans IA.
AUTO_DOWNLOAD_MAX_BYTES = 400 * 1024 * 1024

# Espace disque exige avant d'engager une installation, par poste de depense.
# La pile GPU embarque les runtimes CUDA : elle pese plusieurs fois la pile
# processeur, et le refus doit arriver avant le telechargement, pas au milieu.
DISQUE_REQUIS_VENV_BYTES = 700 * 1024 * 1024
DISQUE_REQUIS_VENV_GPU_BYTES = 4 * 1024 * 1024 * 1024
DISQUE_MARGE_SECURITE_BYTES = 300 * 1024 * 1024

# --- Delais -------------------------------------------------------------------
# Tout processus lance en a un, et le depassement tue le groupe de processus.
DELAI_SONDE_S = 5
DELAI_CREATION_VENV_S = 180
DELAI_PIP_S = 1800
DELAI_TELECHARGEMENT_S = 1800
DELAI_LECTURE_RESEAU_S = 30
DELAI_VALIDATION_GPU_S = 180
DELAI_WORKER_S = 1800

# --- Pipeline -----------------------------------------------------------------
# Format pivot impose entre deux etapes : tableau numpy, RGB, uint8, 0-255.
# Chaque etape convertit ce pivot vers ce dont son modele a besoin, puis y
# revient avant de rendre la main. Un oubli ne leve aucune exception : il
# produit un visage bleu.
PIVOT_ESPACE = "RGB"
PIVOT_TYPE = "uint8"

# Table de preseance des intentions. L'anonymisation annule et remplace toute
# autre modification faciale : faire sourire un visage qu'on vient de flouter
# n'a pas de sens, et l'ordre dans lequel les deux s'appliqueraient tiendrait
# du hasard. L'arbitrage se fait cote greffon, avant l'ecriture du fichier de
# parametres, et il est journalise.
PRESEANCE_FACIALE = ["anonymiser", "sourire", "ameliorer"]

# Delai laisse au pilote pour desallouer reellement apres gc.collect() et la
# destruction d'une session, avant de charger le modele suivant. CUDA utilise
# un allocateur paresseux : sans ce delai, le second modele peut echouer a
# s'instancier alors que la memoire vient d'etre rendue.
PAUSE_ENTRE_MODELES_S = 0.5

# --- Detection des visages ----------------------------------------------------
TAILLE_ENTREE_DETECTION = 640
# YuNet accepte n'importe quelle taille d'entree, mais son cout croit avec
# elle : une photo de douze megapixels y passerait entiere. L'image est donc
# reduite pour que son plus grand cote n'excede pas cette valeur, et les boites
# rendues sont remises a l'echelle de l'image d'origine. C'est une conversion
# de coordonnees de plus, donc une erreur possible de plus : elle est couverte
# par un test geometrique sur le vrai modele.
YUNET_COTE_MAX = 1024
SCORE_MIN_VISAGE = 0.35
NMS_RECOUVREMENT_MAX = 0.45
VISAGES_MAX = 32
# Un visage plus petit que cette fraction du cote de l'image est ignore : en
# dessous, la vignette envoyee aux modeles ne porte plus assez d'information,
# et le resultat est pire que l'original. Le seuil est en cote, pas en aire :
# une hypothese sur l'aire est une hypothese sur la forme.
COTE_MIN_VISAGE_RATIO = 0.01
# Marge ajoutee autour de la boite detectee avant recadrage. Les modeles de
# restauration attendent le menton et le front, que la boite de detection
# coupe.
MARGE_VISAGE_RATIO = 0.35
MARGE_VISAGE_MIN_PX = 12

# --- Anonymisation ------------------------------------------------------------
ANONYME_FLOU = 0
ANONYME_PIXELS = 1
# Rayon du flou gaussien, exprime en fraction du cote de la vignette : un rayon
# fixe laisse un visage lisible sur une photo haute definition.
ANONYME_FLOU_RATIO = 0.25
# Cote de la mosaique, en fraction du cote de la vignette.
ANONYME_PIXELS_RATIO = 0.08

# --- Fusion des vignettes -----------------------------------------------------
# Largeur du fondu applique au bord de la vignette reinseree, en fraction de
# son cote. Hors de la zone fondue, l'ecart pixel avec l'original doit etre
# exactement nul.
FUSION_PLUME_RATIO = 0.12

# --- Amelioration sans IA -----------------------------------------------------
# Repli quand le modele de restauration n'est pas disponible : masque flou puis
# filtre bilateral. Ce n'est pas une restauration, c'est un rehaussement ; le
# nom du calque le dit.
REHAUSSEMENT_FORCE = 0.6
REHAUSSEMENT_RAYON = 3

# --- Journaux et mise au point ------------------------------------------------
VARIABLE_DEBUG = "GIMP_AI_SUITE_DEBUG"
ARCHIVES_A_CONSERVER = 10

# --- Plafond memoire ----------------------------------------------------------
# RLIMIT_AS limite l'espace d'adressage virtuel, pas la memoire residente :
# l'initialisation d'un contexte CUDA reserve couramment des dizaines de
# gigaoctets d'espace virtuel sans les toucher. La limite n'est donc posee que
# lorsqu'aucun fournisseur materiel n'est demande, et elle se dimensionne sur
# la memoire physique reelle.
#
# Aucun Job Object n'est pose sous Windows. La liste de controle exige qu'une
# telle protection soit soit correcte, soit absente, et que sa correction ait
# ete prouvee par un test - pas supposee. Ce depot ne peut pas executer ce
# test, donc la protection est absente et la documentation le dit.
MEMOIRE_PLAFOND_RATIO = 0.75

MARQUEURS_WORKER = (
    "[OK_RESULTAT]",
    "[ERR_PARAMS]",
    "[ERR_IMAGE]",
    "[ERR_IMPORT]",
    "[ERR_MODELE]",
    "[ERR_PIVOT]",
    "[ERR_INFERENCE]",
    "[ERR_AUCUN_VISAGE]",
    "[ERR_ECRITURE]",
    "[ERR_MEMOIRE]",
    "[ERR_INATTENDU]",
    "[INFO_MOTEUR]",
    "[INFO_MATERIEL]",
    "[INFO_DEGRADATION]",
)


# ==============================================================================
# 1b. TABLE DES MODELES
#     Un role par operation. Le contrat d'entree de chaque modele - espace
#     colorimetrique, taille, normalisation, echelle des coordonnees - est une
#     donnee de cette table, jamais une habitude du worker : il se lit dans le
#     code de l'exportateur, et une valeur raisonnable au juge donne un
#     resultat faux sans lever la moindre erreur.
#
#     La normalisation s'ecrit toujours de la meme facon : l'entree passe par
#     x / 255, puis (x - moyenne) / ecart. (0.5, 0.5) redonne x / 127.5 - 1,
#     (0.0, 1.0) redonne x / 255.
#
#     Les adresses et les tailles sont DECLAREES : elles n'ont pas pu etre
#     verifiees depuis ce depot, dont le reseau ne joint pas les hebergeurs de
#     modeles. Une adresse morte ne bloque rien - l'operation degrade et le
#     motif s'affiche - et le fichier peut etre depose a la main, ou une autre
#     adresse indiquee dans sources_modeles.json.
# ==============================================================================
FICHIER_SOURCES_UTILISATEUR = "sources_modeles.json"

ROLE_DETECTION = "detection"
ROLE_DETECTION_YUNET = "detection_yunet"
ROLE_SOURIRE = "sourire"
ROLE_AMELIORATION = "amelioration"
ROLE_COLORISATION = "colorisation"

MODELES = {
    # Detecteur par defaut : YuNet, du zoo de modeles d'OpenCV. 227 Ko, licence
    # Apache-2.0, et surtout OpenCV sait le piloter lui-meme par
    # cv2.FaceDetectorYN - il n'y a donc ni mise en lettre-boite ni decodage de
    # sortie a ecrire, c'est-a-dire ni l'un ni l'autre a se tromper.
    #
    # L'adresse et la taille de ce modele sont MESUREES : le fichier a ete
    # telecharge et charge par OpenCV 4.14 depuis ce depot le 2026-09-17.
    # C'est le seul modele de cette table dont ce soit le cas, et c'est
    # pourquoi c'est le seul que le greffon va chercher de lui-meme.
    #
    # Attention a l'adresse : raw.githubusercontent.com ne rend que le
    # pointeur Git LFS de 131 octets, que le controle de vraisemblance rejette
    # a juste titre. C'est media.githubusercontent.com qui sert le contenu.
    ROLE_DETECTION_YUNET: {
        "fichier": "face_detection_yunet_2023mar.onnx",
        "taille_declaree": 232589,
        "taille_min": 100 * 1024,
        "source_verifiee": True,
        "sources": [
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        "taille_entree": YUNET_COTE_MAX,
        "normalisation": [0.0, 1.0],
        "espace": "BGR",
    },
    # Detecteur alternatif. Aucune adresse n'est declaree : les deux qui
    # figuraient ici renvoyaient HTTP 401 chez l'utilisateur, le depot ayant
    # disparu ou etant devenu prive. Une adresse dont on sait qu'elle ne
    # repond pas ne vaut pas mieux que pas d'adresse du tout, et elle coute
    # une requete a chaque lancement. Ce modele s'obtient donc par depot
    # manuel, ou par une adresse declaree dans sources_modeles.json - et il
    # passe alors devant YuNet, parce qu'un fichier depose l'a ete
    # deliberement.
    ROLE_DETECTION: {
        "fichier": "yolov8n-face.onnx",
        "taille_declaree": 13 * 1024 * 1024,
        "sources": [],
        "taille_entree": TAILLE_ENTREE_DETECTION,
        "normalisation": [0.0, 1.0],
        "espace": "RGB",
    },
    ROLE_SOURIRE: {
        "fichier": "attgan_smile.onnx",
        "taille_declaree": 150 * 1024 * 1024,
        "sources": [],
        "taille_entree": 256,
        "normalisation": [0.5, 0.5],
        "espace": "RGB",
    },
    ROLE_AMELIORATION: {
        "fichier": "gfpgan_1_4.onnx",
        "taille_declaree": 340 * 1024 * 1024,
        # Cette adresse renvoyait HTTP 404 chez l'utilisateur le 2026-09-17,
        # comme les deux du detecteur avant elle. Retiree pour la meme raison.
        # Les poids officiels .pth, eux, restent joignables sur les Releases
        # GitHub de GFPGAN : la conversion en ONNX est une piste, pas une
        # adresse.
        "sources": [],
        "taille_entree": 512,
        "normalisation": [0.5, 0.5],
        "espace": "RGB",
    },
    ROLE_COLORISATION: {
        "fichier": "deoldify_artistic.onnx",
        "taille_declaree": 250 * 1024 * 1024,
        "sources": [],
        "taille_entree": 256,
        "normalisation": [0.5, 0.5],
        "espace": "RGB",
    },
}

# Roles dont l'absence se rattrape par un chemin sans IA, et roles dont
# l'absence annule simplement l'operation. La distinction gouverne le message
# affiche : un repli s'annonce, une operation annulee s'explique.
ROLES_AVEC_REPLI = (ROLE_DETECTION, ROLE_DETECTION_YUNET,
                    ROLE_AMELIORATION)

# Vecteur d'attributs du modele de sourire. Sa taille et l'indice de
# l'attribut "sourire" dependent de l'export : ils sont transmis au worker,
# jamais devines par lui.
SOURIRE_VECTEUR_TAILLE = 13
SOURIRE_INDICE_ATTRIBUT = 12

# Fournisseurs onnxruntime demandes quand la case "carte graphique" est cochee.
# La liste est croisee cote worker avec les fournisseurs reellement disponibles :
# le moteur refuse une demande qu'il ne peut pas satisfaire.
FOURNISSEURS_GPU = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]
FOURNISSEURS_CPU = ["CPUExecutionProvider"]


def annonce_cout(role):
    """Ce que coute reellement une option, derive de la table des modeles.

    Un libelle ecrit a la main derive : celui du sourire annoncait "environ
    150 Mo telecharges au premier usage" alors qu'aucune adresse n'etait
    declaree pour ce modele - la case promettait donc un telechargement que le
    code ne pouvait pas faire, et l'utilisateur ne l'apprenait qu'apres coup.
    Le libelle se calcule ici, a partir de la seule chose qui fasse foi : la
    presence ou non d'une source.
    """
    entree = MODELES[role]
    taille = octets_lisibles(entree.get("taille_declaree", 0))
    if not entree["sources"]:
        return ("modele a fournir : deposez %s dans le dossier des modeles, le "
                "greffon n'a pas d'adresse pour le telecharger"
                % entree["fichier"])
    if entree.get("source_verifiee"):
        return "environ %s telecharges au premier usage" % taille
    # Distinguer ce qui est mesure de ce qui est declare, jusque dans
    # l'interface : une adresse que personne n'a jointe depuis ce depot ne
    # doit pas etre annoncee comme un telechargement acquis.
    return ("telechargement d'environ %s tente au premier usage ; l'adresse "
            "n'a pas ete verifiee, et le greffon degrade si elle ne repond pas"
            % taille)


def octets_lisibles(n):
    """Taille en Mo ou Go, bornee et sans fausse precision."""
    try:
        n = float(n)
    except Exception:
        return "taille inconnue"
    if n <= 0:
        return "taille inconnue"
    if n < 1024 * 1024:
        return "%d Ko" % max(1, int(n / 1024))
    if n < 1024 * 1024 * 1024:
        return "%.1f Mo" % (n / (1024.0 * 1024.0))
    return "%.2f Go" % (n / (1024.0 * 1024.0 * 1024.0))


# ==============================================================================
# 2. EMPLACEMENTS ET JOURNAL DE DIAGNOSTIC
#    Fichiers legers sous Gimp.directory(), donnees lourdes ailleurs : sous
#    Windows, Gimp.directory() vit dans AppData\Roaming, synchronise a chaque
#    ouverture et fermeture de session sur un profil itinerant. Plusieurs
#    gigaoctets de venv et de modeles y deviennent un incident d'exploitation.
#
#    Ressources partagees avec le reste de la suite - c'est le conflit le plus
#    couteux d'une suite de greffons, et il n'existe qu'a partir du deuxieme :
#
#      venv-<pile>/            partage entre greffons de meme pile
#      models/                 partage, indexe par nom de fichier
#      trusted_models.json     partage, indexe par nom de fichier
#      interpreteur_<pile>.json partage, simple cache
#      env_<pile>_<greffon>.json   PAR GREFFON, et c'est volontaire
#
#    Le marqueur d'environnement porte le nom du greffon parce qu'il contient
#    sa version : un marqueur commun ferait qu'a chaque bascule entre deux
#    greffons de versions differentes, chacun jugerait l'environnement perime
#    et le reinstallerait, indefiniment et sans le moindre message.
# ==============================================================================
LEGACY_VENV_DIR_NAMES = ["venv_onnx", "venv"]
LEGACY_CACHE_NAMES = ["python_interpreter_v2.json", "python_interpreter.json"]

_DOSSIER_DONNEES = None
_PILE_ACTIVE = STACK_CPU


def pile_active():
    return _PILE_ACTIVE


def definir_pile(pile):
    global _PILE_ACTIVE
    _PILE_ACTIVE = pile


def paquets_de_pile(pile):
    return REQUIRED_PACKAGES_GPU if pile == STACK_GPU else REQUIRED_PACKAGES


def nom_venv(pile):
    return "venv-" + pile


def nom_marqueur(pile):
    return "env_" + pile + "_" + PLUGIN_ID + ".json"


def nom_cache_interpreteur(pile):
    return "interpreteur_" + pile + ".json"


def dossier_memorise():
    """Dossier de donnees retenu lors d'une installation precedente.

    Il est note dans le marqueur, qui vit avec le profil GIMP. Si la racine du
    systeme a change - profil deplace, LOCALAPPDATA redirige, lettre de lecteur
    differente - mais que l'ancien dossier existe toujours, autant le reprendre
    que retelecharger.
    """
    try:
        chemin = lire_marqueur().get("dossier_donnees")
    except Exception:
        return None
    if chemin and os.path.isdir(chemin):
        return chemin
    return None


def get_data_dir():
    """Dossier des donnees volumineuses : environnement Python et modeles.

    Il ne porte volontairement pas de numero de version de GIMP. Les poids ONNX
    et le venv ne dependent pas de la version de GIMP : les indexer par version
    ferait retelecharger plusieurs centaines de megaoctets a chaque mise a
    jour, et laisserait l'ancien dossier immobilise sans que personne ne le
    remarque.

    Ordre de resolution : la variable d'environnement si elle est posee, puis
    l'emplacement canonique, puis le dossier memorise par une installation
    precedente, puis un dossier versionne a migrer par simple renommage.
    """
    global _DOSSIER_DONNEES
    if _DOSSIER_DONNEES:
        return _DOSSIER_DONNEES

    force = os.environ.get(VARIABLE_DOSSIER, "").strip()
    if force:
        try:
            os.makedirs(os.path.join(force, "models"), exist_ok=True)
            _DOSSIER_DONNEES = force
            journal("dossier de donnees impose par %s: %s" % (VARIABLE_DOSSIER, force))
            return force
        except Exception as e:
            journal("dossier impose inutilisable (%s), retour aux conventions: %s"
                    % (e, force))

    base = base_donnees()
    cible = os.path.join(base, "GIMP", SHARED_DIR_NAME)
    if not os.path.isdir(cible):
        memorise = dossier_memorise()
        if memorise:
            journal("dossier de donnees repris du marqueur: " + memorise)
            cible = memorise
        else:
            racine = os.path.join(base, "GIMP")
            anciens = []
            try:
                for entree in sorted(os.listdir(racine)):
                    if entree == SHARED_DIR_NAME:
                        continue
                    candidat = os.path.join(racine, entree, SHARED_DIR_NAME)
                    if os.path.isdir(candidat):
                        anciens.append(candidat)
            except Exception:
                anciens = []
            if anciens:
                ancien = anciens[0]
                try:
                    os.rename(ancien, cible)
                    journal("dossier de donnees migre: %s -> %s" % (ancien, cible))
                except Exception as e:
                    journal("migration du dossier de donnees impossible (%s), "
                            "reprise sur place: %s" % (e, ancien))
                    cible = ancien

    try:
        os.makedirs(os.path.join(cible, "models"), exist_ok=True)
    except Exception:
        pass
    _DOSSIER_DONNEES = cible
    return cible


def get_models_dir():
    return os.path.join(get_data_dir(), "models")


def get_shared_dir():
    try:
        shared = os.path.join(Gimp.directory(), SHARED_DIR_NAME)
    except Exception:
        shared = os.path.join(os.path.expanduser("~"), "." + SHARED_DIR_NAME)
    try:
        os.makedirs(shared, exist_ok=True)
    except Exception:
        pass
    return shared


def get_logs_dir():
    d = os.path.join(get_shared_dir(), "logs")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def get_marker_path():
    return os.path.join(get_shared_dir(), nom_marqueur(pile_active()))


def get_interpreter_cache():
    return os.path.join(get_shared_dir(), nom_cache_interpreteur(pile_active()))


def get_tofu_path():
    # Indexe par nom de fichier : partage sans risque entre greffons.
    return os.path.join(get_shared_dir(), "trusted_models.json")


def get_inventory_path():
    return os.path.join(get_shared_dir(), "inventaire_" + PLUGIN_ID + ".json")


def get_sources_utilisateur_path():
    return os.path.join(get_shared_dir(), FICHIER_SOURCES_UTILISATEUR)


def journal(message):
    """Trace sur disque, ecrite des la premiere ligne de do_query_procedures.

    Un echec d'enregistrement de greffon est silencieux par construction : GIMP
    n'expose pas la sortie d'erreur en mode graphique. L'absence de ce fichier
    est elle-meme une information : elle prouve que le fichier n'a pas ete
    execute, et oriente vers le transport plutot que vers le code.
    """
    try:
        path = os.path.join(get_shared_dir(), "journal_" + PLUGIN_ID + ".log")
        ligne = "%s v%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                 PLUGIN_VERSION, message)
        with open(path, "a", encoding="utf-8") as f:
            f.write(ligne)
    except Exception:
        pass


def mode_debug():
    return bool(os.environ.get(VARIABLE_DEBUG, "").strip())


def sources_du_modele(role):
    """Adresses candidates d'un modele : la table du code, completee par le
    fichier sources_modeles.json du dossier partage.

    Ce fichier n'est jamais necessaire : il existe pour qu'une adresse devenue
    morte se repare sans toucher au code, ce que ce greffon ne demandera
    jamais. Les entrees de l'utilisateur passent devant.
    """
    sources = []
    try:
        with open(get_sources_utilisateur_path(), "r", encoding="utf-8",
                  errors="replace") as f:
            donnees = json.load(f)
        valeurs = donnees.get(role) if isinstance(donnees, dict) else None
        if isinstance(valeurs, str):
            valeurs = [valeurs]
        for url in (valeurs or []):
            if isinstance(url, str) and url.strip().lower().startswith("http"):
                sources.append(url.strip())
    except Exception:
        pass
    for url in MODELES[role]["sources"]:
        if url not in sources:
            sources.append(url)
    return sources


# ==============================================================================
# 3. ISOLATION D'ENVIRONNEMENT ET CONDUITE DES PROCESSUS
# ==============================================================================
# Variables qui font charger a un Python systeme les bibliotheques de GIMP. Un
# Python lance en les heritant charge la libstdc++ ou la libpng de GIMP, et
# l'import echoue avec un message incomprehensible.
VARIABLES_A_PURGER = [
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT",
    "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_INSERT_LIBRARIES",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "GI_TYPELIB_PATH", "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR",
    "GSETTINGS_SCHEMA_DIR", "GEGL_PATH", "BABL_PATH",
]

# Processus enfants encore vivants. Le worker ne doit jamais survivre a son
# parent : ce registre garantit qu'il est tue quoi qu'il arrive, y compris sur
# une exception qui ne passe par aucune boucle d'attente.
_PROCESSUS_VIVANTS = []


def clean_env(dossiers_dll=None):
    """Environnement d'execution des processus enfants.

    Le filtrage du PATH reste chirurgical : on ne retire que les entrees
    contenant "gimp". Un venv n'herite pas du site-packages du systeme, mais il
    herite du PATH, donc des bibliotheques dynamiques qui s'y trouvent - un
    runtime CUDA installe a l'echelle de la machine, par exemple. Une purge
    plus large couperait l'acceleration materielle sans que rien ne l'explique.

    dossiers_dll : dossiers a exposer en tete de PATH. Les roues pip de cuDNN
    installent leurs bibliotheques dans le site-packages, ou le chargeur
    dynamique ne va pas les chercher : sans cette exposition, elles sont
    installees mais introuvables.
    """
    env = os.environ.copy()
    for nom in VARIABLES_A_PURGER:
        env.pop(nom, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # PYTHONIOENCODING seul ne suffit pas : la sortie des bibliotheques
    # natives ne suit pas celle de Python.
    env["PYTHONUTF8"] = "1"

    try:
        gimp_py = os.path.dirname(os.path.abspath(sys.executable)).lower()
    except Exception:
        gimp_py = ""
    gardees = []
    for p in env.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        bas = p.lower()
        if "gimp" in bas:
            continue
        try:
            if gimp_py and os.path.abspath(p).lower() == gimp_py:
                continue
        except Exception:
            pass
        gardees.append(p)
    for dossier in reversed(dossiers_dll or []):
        if dossier and os.path.isdir(dossier):
            gardees.insert(0, dossier)
    env["PATH"] = os.pathsep.join(gardees)
    return env


def flatpak_detecte():
    return os.path.exists("/.flatpak-info")


def flags_creation():
    # CREATE_NO_WINDOW, toujours conditionne par la plateforme.
    return 0x08000000 if os.name == "nt" else 0


def demarrer_processus(cmd, env, fichier_log=None):
    """Popen avec sortie sur fichier (jamais sur PIPE) et groupe de processus.

    Un pipe non consomme se bloque des que le tampon OS est plein pendant que
    la boucle d'attente n'appelle que poll(). Le piege se manifeste avant meme
    l'inference : un pip install d'une pile scientifique produit plusieurs
    centaines de kilooctets la ou le tampon POSIX est de 64 Ko.
    """
    kwargs = {"env": env, "creationflags": flags_creation()}
    if os.name != "nt":
        kwargs.pop("creationflags")
        # Groupe de processus propre : un worker essaime des sous-processus
        # qu'un simple kill sur le PID direct laisse orphelins.
        kwargs["start_new_session"] = True
    if fichier_log is not None:
        kwargs["stdout"] = fichier_log
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    proc = subprocess.Popen(cmd, **kwargs)
    _PROCESSUS_VIVANTS.append(proc)
    return proc


def tuer_groupe(proc):
    """Tue le processus et toute sa descendance."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=20,
                           creationflags=flags_creation())
        else:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def tuer_processus_restants():
    """Filet de securite appele dans le finally du traitement.

    Le worker ne doit jamais survivre a son parent : sans cela, un abandon
    laisse un processus saturer le processeur en arriere-plan jusqu'a la fin
    de son calcul, longtemps apres que GIMP a rendu la main.
    """
    restants = 0
    for proc in list(_PROCESSUS_VIVANTS):
        try:
            if proc.poll() is None:
                tuer_groupe(proc)
                restants += 1
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _PROCESSUS_VIVANTS.remove(proc)
        except ValueError:
            pass
    if restants:
        journal("%d processus enfant(s) tue(s) a la sortie" % restants)
    return restants


# Sondes d'annulation essayees une seule fois, dans l'ordre. Aucune n'est
# garantie : l'API de GIMP 3.0 n'expose pas de facon stable, pour un greffon,
# l'etat du bouton Annuler de la barre de progression. Le greffon tente donc
# ce qui existe, et la documentation dit franchement que la detection peut ne
# pas etre disponible sur une build donnee. Ce qui reste garanti quoi qu'il
# arrive, c'est que le processus enfant est tue quand la boucle s'arrete.
NOMS_SONDES_ANNULATION = (
    "progress_get_cancelled",
    "progress_is_cancelled",
    "progress_cancelled",
)
_SONDE_ANNULATION = False
_SONDE_ANNULATION_FAITE = False


def sonde_annulation():
    global _SONDE_ANNULATION, _SONDE_ANNULATION_FAITE
    if _SONDE_ANNULATION_FAITE:
        return _SONDE_ANNULATION
    _SONDE_ANNULATION_FAITE = True
    _SONDE_ANNULATION = None
    for nom in NOMS_SONDES_ANNULATION:
        fonction = getattr(Gimp, nom, None)
        if fonction is None:
            continue
        try:
            fonction()
        except Exception:
            continue
        _SONDE_ANNULATION = fonction
        journal("sonde d'annulation disponible: Gimp." + nom)
        break
    if _SONDE_ANNULATION is None:
        journal("aucune sonde d'annulation exposee par cette build de GIMP ; "
                "le processus enfant reste tue a la sortie de la boucle")
    return _SONDE_ANNULATION


def annulation_demandee():
    sonde = sonde_annulation()
    if sonde is None:
        return False
    try:
        return bool(sonde())
    except Exception:
        return False


def attendre_processus(proc, delai_s, texte_progression=None):
    """Attente animee et bornee. Retourne (code_retour, depassement, annule).

    Jamais subprocess.run() pour un traitement long : une boucle de polling
    anime la barre de progression, et surtout elle permet d'interrompre. Une
    boucle sans borne gelerait GIMP indefiniment si le processus se bloque.
    """
    debut = time.time()
    while proc.poll() is None:
        if time.time() - debut > delai_s:
            tuer_groupe(proc)
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            return (proc.returncode, True, False)
        if annulation_demandee():
            journal("annulation demandee par l'utilisateur")
            tuer_groupe(proc)
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            return (proc.returncode, False, True)
        try:
            Gimp.progress_pulse()
            if texte_progression:
                Gimp.progress_set_text(texte_progression)
        except Exception:
            pass
        time.sleep(0.2)
    try:
        _PROCESSUS_VIVANTS.remove(proc)
    except ValueError:
        pass
    return (proc.returncode, False, False)


def lire_journal(path):
    """Relecture tolerante d'un journal.

    La sortie des bibliotheques natives ne suit pas l'encodage de Python : une
    relecture en UTF-8 strict rendrait la moitie du journal illisible, donc
    inexploitable, alors que c'est precisement son role.
    """
    try:
        with open(path, "rb") as f:
            brut = f.read()
    except Exception:
        return ""
    if not brut:
        return ""
    essais = []
    if b"\x00" in brut[:64]:
        essais.append("utf-16")
    essais.extend(["utf-8", "cp1252", "latin-1"])
    for enc in essais:
        try:
            return brut.decode(enc)
        except Exception:
            continue
    return brut.decode("utf-8", "replace")


# ==============================================================================
# 4. DECOUVERTE DE L'INTERPRETEUR PYTHON
#    Classement par canal de decouverte d'abord, par version ensuite. Un Python
#    livre avec une application tierce (Blender, Inkscape...) disparait a la
#    mise a jour de cette application et emporte l'environnement avec lui : il
#    est declasse, jamais rejete, car sur un poste qui n'a rien d'autre,
#    refuser revient a ne pas fonctionner.
# ==============================================================================
CANAL_REGISTRE = ("registre", 90)
CANAL_LANCEUR = ("lanceur_py", 90)
CANAL_STANDARD = ("emplacement_standard", 70)
CANAL_VENV_CONNU = ("venv_connu", 50)
CANAL_PATH = ("path", 30)

PENALITE_APPLICATION_TIERCE = 60
APPLICATIONS_TIERCES = [
    "blender", "gimp", "inkscape", "libreoffice", "openoffice", "qgis",
    "resolve", "krita", "darktable", "houdini", "maya", "nuke", "unity",
    "unreal", "msys", "mingw", "cygwin", "windowsapps",
]

# Sortie reelle du lanceur, capturee sur un poste Windows :
#   -V:3.14 *        C:\Users\x\AppData\Local\Programs\Python\Python314\python.exe
# Le chemin suit le tag de version, separe par des espaces : un split()[-1]
# renvoie "Files\PythonXX\python.exe" des que le chemin contient "Program
# Files". L'extraction se fait donc par expression reguliere ancree sur la
# lettre de lecteur. Le motif est couvert par outils/tests_unitaires.py a
# partir de la sortie capturee dans outils/sorties_reelles/.
MOTIF_LANCEUR_PY = r"([a-zA-Z]:\\[^\r\n]*?python(?:[0-9._]*)\.exe)"


def sonder_interpreteur(exe, env):
    """Teste reellement le candidat par execution. Retourne la version ou None.

    La decision repose sur le code de retour ; la version est ensuite lue dans
    un JSON, ce qui est une extraction de donnee et non un constat de reussite.
    """
    if not exe or not os.path.isfile(exe):
        return None
    if "gimp" in exe.lower():
        return None
    script = (
        "import sys, json, venv, ensurepip;"
        "sys.stdout.write(json.dumps(list(sys.version_info[:3])));"
        "sys.exit(0)"
    )
    try:
        res = subprocess.run([exe, "-c", script], capture_output=True,
                             timeout=DELAI_SONDE_S, env=env,
                             creationflags=flags_creation())
    except Exception:
        return None
    if res.returncode != 0:
        return None
    try:
        brut = res.stdout.decode("utf-8", "replace").strip()
        version = tuple(int(x) for x in json.loads(brut))
    except Exception:
        return None
    if len(version) < 3:
        return None
    return version


def _candidats_windows(env, ajouter):
    import re
    # 1. Lanceur py officiel.
    try:
        res = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                             timeout=DELAI_SONDE_S, env=env,
                             creationflags=flags_creation())
        for chemin in re.findall(MOTIF_LANCEUR_PY, res.stdout or ""):
            ajouter(chemin.strip(), CANAL_LANCEUR)
    except Exception:
        pass

    # 2. Registre : deux ruches, deux arborescences, deux valeurs par version.
    ruches = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
    arbres = [r"Software\Python\PythonCore", r"Software\WOW6432Node\Python\PythonCore"]
    vues = [0]
    for nom_vue in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
        vues.append(getattr(winreg, nom_vue, 0))
    for ruche in ruches:
        for arbre in arbres:
            for vue in vues:
                try:
                    acces = winreg.KEY_READ | vue
                    with winreg.OpenKey(ruche, arbre, 0, acces) as rk:
                        i = 0
                        while True:
                            try:
                                version = winreg.EnumKey(rk, i)
                            except OSError:
                                break
                            i += 1
                            sous = arbre + "\\" + version + "\\InstallPath"
                            try:
                                with winreg.OpenKey(ruche, sous, 0, acces) as ipk:
                                    for nom_valeur in ("ExecutablePath", ""):
                                        try:
                                            val, _ = winreg.QueryValueEx(ipk, nom_valeur)
                                        except OSError:
                                            continue
                                        if not val:
                                            continue
                                        if nom_valeur == "":
                                            val = os.path.join(val, "python.exe")
                                        ajouter(val, CANAL_REGISTRE)
                            except OSError:
                                pass
                except Exception:
                    pass

    # 3. Emplacements d'installation standards. Balayage borne en profondeur :
    #    un os.walk complet de Program Files coute des minutes.
    racines = [
        os.path.expandvars(r"%LocalAppData%\Programs\Python"),
        os.path.expandvars(r"%ProgramFiles%"),
        os.path.expandvars(r"%ProgramFiles(x86)%"),
        r"C:\\",
    ]
    for racine in racines:
        if not racine or not os.path.isdir(racine):
            continue
        try:
            for entree in os.listdir(racine):
                if not entree.lower().startswith("python"):
                    continue
                ajouter(os.path.join(racine, entree, "python.exe"), CANAL_STANDARD)
        except Exception:
            pass

    # 4. PATH : dernier recours.
    for p in env.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        ajouter(os.path.join(p, "python.exe"), CANAL_PATH)


def _candidats_posix(env, ajouter):
    for nom in ("python3", "python"):
        try:
            res = subprocess.run(["which", "-a", nom], capture_output=True,
                                 text=True, timeout=DELAI_SONDE_S, env=env)
            for ligne in (res.stdout or "").splitlines():
                if ligne.strip():
                    ajouter(ligne.strip(), CANAL_PATH)
        except Exception:
            pass
    for base in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"):
        if os.path.isdir(base):
            try:
                for entree in os.listdir(base):
                    if entree.startswith("python3"):
                        ajouter(os.path.join(base, entree), CANAL_STANDARD)
            except Exception:
                pass
    home = os.path.expanduser("~")
    for v in (".venvs", ".virtualenvs", ".venv"):
        racine = os.path.join(home, v)
        if not os.path.isdir(racine):
            continue
        try:
            for entree in os.listdir(racine):
                ajouter(os.path.join(racine, entree, "bin", "python3"), CANAL_VENV_CONNU)
        except Exception:
            pass


def classer_candidat(chemin, canal):
    """Score de canal, penalise si l'executable appartient a une application
    tierce (un dossier ancetre contient son nom)."""
    nom_canal, score = canal
    bas = os.path.normpath(chemin).lower()
    for app in APPLICATIONS_TIERCES:
        if app in bas:
            return (nom_canal + "+application_tierce", max(1, score - PENALITE_APPLICATION_TIERCE))
    return (nom_canal, score)


def trier_candidats(candidats):
    """Canal de decouverte d'abord, version ensuite.

    Retenir le premier interpreteur qui repond, sans regarder d'ou il vient ni
    quelle version il porte, fait echouer l'installation bien plus loin, avec
    un message de pip incomprehensible, sur tout poste ou cohabitent plusieurs
    Python.
    """
    return sorted(candidats,
                  key=lambda d: (d.get("score_canal", 0), d.get("dans_plafond", 0),
                                 d.get("version", (0,))),
                  reverse=True)


def decouvrir_pythons():
    """Retourne une liste de dict tries : le meilleur candidat en tete."""
    env = clean_env()
    vus = {}

    def ajouter(chemin, canal):
        try:
            norm = os.path.normpath(chemin)
        except Exception:
            return
        if not norm or not os.path.isfile(norm):
            return
        nom_canal, score = classer_candidat(norm, canal)
        cle = norm.lower()
        if cle not in vus or vus[cle]["score_canal"] < score:
            vus[cle] = {"chemin": norm, "canal": nom_canal, "score_canal": score}

    if os.name == "nt":
        _candidats_windows(env, ajouter)
    else:
        _candidats_posix(env, ajouter)

    valides = []
    for info in vus.values():
        version = sonder_interpreteur(info["chemin"], env)
        if version is None:
            continue
        if version < PY_MIN:
            # Sous le plancher, l'explication est encore possible : on ecarte.
            continue
        info["version"] = version
        # Le plafond n'interdit rien, il classe : une version testee passe
        # devant une version plus recente que la derniere campagne de test.
        info["dans_plafond"] = 1 if version <= PY_MAX_TESTED else 0
        valides.append(info)

    valides = trier_candidats(valides)
    journal("interpreteurs retenus: " + ", ".join(
        "%s (%s, %s)" % (d["chemin"], d["canal"], ".".join(str(x) for x in d["version"]))
        for d in valides[:4]) if valides else "aucun interpreteur valide")
    return valides



# ==============================================================================
# 5. MARQUEUR D'ENVIRONNEMENT, PILES ET ACCELERATION MATERIELLE
#    Le marqueur evite un import complet a chaque lancement (deux a cinq
#    secondes avant le debut du travail utile). Il consigne le canal de
#    decouverte, l'horodatage, la version de l'API obtenue et le materiel
#    reellement constate : un marqueur qui n'enregistre que le chemin ne
#    laisse, trois mois plus tard, qu'un chemin orphelin et des hypotheses.
# ==============================================================================
def signature_paquets(pile=None):
    pile = pile or pile_active()
    brut = "|".join(sorted(paquets_de_pile(pile))) + "|" + pile
    return hashlib.sha256(brut.encode("utf-8")).hexdigest()[:16]


def lire_marqueur():
    try:
        with open(get_marker_path(), "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def ecrire_marqueur(donnees):
    donnees = dict(donnees)
    donnees["version_greffon"] = PLUGIN_VERSION
    donnees["api_gimp"] = API_GIMP
    donnees["pile"] = pile_active()
    try:
        # setdefault : une valeur transmise par l'appelant fait foi. Ecraser
        # ici le dossier memorise reviendrait a perdre l'emplacement d'origine
        # au premier enregistrement fait depuis une autre racine.
        donnees.setdefault("dossier_donnees", _DOSSIER_DONNEES or get_data_dir())
    except Exception:
        pass
    donnees["signature_paquets"] = signature_paquets()
    donnees["ecrit_le"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with open(get_marker_path(), "w", encoding="utf-8") as f:
            json.dump(donnees, f, indent=2)
    except Exception:
        pass
    return donnees


def marqueur_utilisable(marqueur):
    if not marqueur:
        return False
    if marqueur.get("statut") != "pret":
        return False
    if marqueur.get("signature_paquets") != signature_paquets():
        return False
    if marqueur.get("version_greffon") != PLUGIN_VERSION:
        return False
    py = marqueur.get("venv_python")
    return bool(py) and os.path.isfile(py)


def invalider_marqueur(raison):
    marqueur = lire_marqueur()
    marqueur["statut"] = "a_reinstaller"
    marqueur["derniere_erreur"] = str(raison)[:400]
    ecrire_marqueur(marqueur)
    journal("marqueur invalide: " + str(raison)[:200])


def chemin_python_venv(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def migrer_anciens_noms(pile):
    """Reprend un venv et un cache d'interpreteur construits par une version
    anterieure ou par un autre greffon de la meme pile, plutot que d'imposer
    un nouveau telechargement de plusieurs centaines de megaoctets."""
    data_dir = get_data_dir()
    cible = os.path.join(data_dir, nom_venv(pile))
    if not os.path.isdir(cible) and pile == STACK_CPU:
        for ancien in LEGACY_VENV_DIR_NAMES:
            source = os.path.join(data_dir, ancien)
            if os.path.isdir(source) and os.path.isfile(chemin_python_venv(source)):
                try:
                    os.rename(source, cible)
                    journal("venv migre: %s -> %s" % (ancien, nom_venv(pile)))
                except Exception as e:
                    journal("migration venv impossible (%s), reprise sur place" % e)
                    return source
                break
    cache = get_interpreter_cache()
    if not os.path.isfile(cache):
        for ancien in LEGACY_CACHE_NAMES:
            source = os.path.join(get_shared_dir(), ancien)
            if os.path.isfile(source):
                try:
                    shutil.copyfile(source, cache)
                    journal("cache interpreteur migre: " + ancien)
                except Exception:
                    pass
                break
    return cible


def espace_libre(chemin):
    sonde = chemin
    while sonde and not os.path.isdir(sonde):
        parent = os.path.dirname(sonde)
        if parent == sonde:
            break
        sonde = parent
    try:
        return shutil.disk_usage(sonde).free
    except Exception:
        return -1


def verifier_espace(chemin, requis, quoi):
    """Controle engage seulement quand une installation l'est reellement : un
    environnement deja complet doit rester utilisable sur un disque plein.

    L'alternative est un "No space left on device" au milieu d'un
    telechargement, qui laisse en plus un environnement a moitie peuple.
    """
    libre = espace_libre(chemin)
    if libre < 0:
        return
    if libre < requis + DISQUE_MARGE_SECURITE_BYTES:
        raise RuntimeError(
            "Espace disque insuffisant pour %s.\n"
            "Requis : %s (dont %s de marge)\n"
            "Disponible : %s\n"
            "Chemin : %s" % (quoi,
                             octets_lisibles(requis + DISQUE_MARGE_SECURITE_BYTES),
                             octets_lisibles(DISQUE_MARGE_SECURITE_BYTES),
                             octets_lisibles(libre), chemin))


# ------------------------------------------------------------------------------
# Detection du materiel : sonder la couche la plus basse dont la pile a
# reellement besoin, et rien en dessous.
#
# Les roues de PyTorch embarquent leur propre runtime CUDA : sonder le pilote y
# suffirait. onnxruntime-gpu ne l'embarque pas et reclame un CUDA Toolkit
# installe sur le systeme. Sonder le pilote ferait donc basculer vers la roue
# GPU tous les postes qui n'ont qu'un pilote - soit la grande majorite - pour
# un repli silencieux sur le processeur apres plusieurs gigaoctets telecharges.
# C'est le runtime qu'on sonde ici, jamais le pilote.
# ------------------------------------------------------------------------------
_RUNTIME_CUDA = None


def sonde_runtime_cuda():
    """Retourne (present, detail). Le detail est conserve pour le diagnostic,
    pas seulement le booleen : un message qui ne dirait que "non" ne servirait
    a rien le jour ou il faut comprendre pourquoi."""
    global _RUNTIME_CUDA
    if _RUNTIME_CUDA is not None:
        return _RUNTIME_CUDA

    indices = []
    trouve = False

    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path and os.path.isdir(cuda_path):
        indices.append("CUDA_PATH=" + cuda_path)
        trouve = True
    elif cuda_path:
        indices.append("CUDA_PATH pointe un dossier absent (%s)" % cuda_path)

    if not trouve:
        try:
            from ctypes.util import find_library
            for nom in ("cudart", "cudart64_12", "cudart64_11", "cudart64_110"):
                chemin = find_library(nom)
                if chemin:
                    indices.append("cudart trouve: " + str(chemin))
                    trouve = True
                    break
        except Exception as e:
            indices.append("find_library indisponible: %s" % e)

    if not trouve and os.name != "nt":
        for racine in ("/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"):
            if os.path.isdir(racine):
                indices.append("toolkit present: " + racine)
                trouve = True
                break

    if not indices:
        indices.append("aucun CUDA Toolkit ni runtime cudart trouve")
    _RUNTIME_CUDA = (trouve, " ; ".join(indices))
    journal("sonde runtime CUDA: %s (%s)" % (trouve, _RUNTIME_CUDA[1]))
    return _RUNTIME_CUDA


def dossiers_dll_paquets(python_venv):
    """Dossiers de bibliotheques des roues nvidia installees dans le venv.

    Elles sont posees dans le site-packages, ou le chargeur dynamique ne va
    pas les chercher : installees mais introuvables sans cette exposition.
    """
    dossiers = []
    racine = os.path.dirname(os.path.dirname(os.path.abspath(python_venv)))
    candidats = []
    if os.name == "nt":
        candidats.append(os.path.join(racine, "Lib", "site-packages", "nvidia"))
    else:
        lib = os.path.join(racine, "lib")
        try:
            for entree in os.listdir(lib):
                candidats.append(os.path.join(lib, entree, "site-packages", "nvidia"))
        except Exception:
            pass
    for base in candidats:
        if not os.path.isdir(base):
            continue
        for dossier, _, _ in os.walk(base):
            nom = os.path.basename(dossier).lower()
            if nom in ("bin", "lib"):
                dossiers.append(dossier)
    return dossiers


def valider_venv(py, env, pile=None):
    """Verification reelle par import, utilisee seulement quand le marqueur ne
    permet pas de conclure.

    La sonde porte sur l'import du module, jamais sur un nom de distribution :
    "pip show onnxruntime" repond "non trouve" sur une machine ou le module est
    parfaitement importable, fourni par onnxruntime-gpu.
    """
    check = "import numpy, cv2, onnxruntime; print(onnxruntime.__version__)"
    try:
        res = subprocess.run([py, "-c", check], capture_output=True,
                             timeout=120, env=env, creationflags=flags_creation())
        return res.returncode == 0
    except Exception:
        return False


# Validation de l'acceleration par une inference reelle. Interroger la
# bibliotheque ne prouve rien : onnxruntime repond que CUDAExecutionProvider
# est disponible meme lorsque cuDNN est absent. La session se cree, le marqueur
# enregistre un environnement "pret", et l'echec ne survient qu'au premier
# noeud de convolution - c'est-a-dire au milieu du travail de l'utilisateur.
# Seule une inference reelle sur une petite entree valide un chemin
# d'acceleration. Elle coute quelques secondes, une fois, a l'installation.
VALIDATION_SCRIPT = r'''# -*- coding: utf-8 -*-

import json
import sys


def main():
    if len(sys.argv) < 2:
        return 2
    with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
        cfg = json.load(f)
    sortie = cfg.get("sortie")
    resultat = {"ok": False, "fournisseur": "", "detail": "",
                "disponibles": [], "demandes": [], "version": ""}
    try:
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False

        # Croiser la demande avec les fournisseurs reellement disponibles,
        # exactement comme le fait le worker. Sans ce croisement, un seul nom
        # inconnu de cette build - ROCMExecutionProvider sur une machine
        # Windows, par exemple - fait rejeter la liste ENTIERE par le moteur,
        # qui se rabat alors sur le processeur. L'acceleration etait donc
        # declaree impossible sur des postes ou elle fonctionnait, et le
        # message accusait cuDNN.
        disponibles = list(ort.get_available_providers())
        resultat["disponibles"] = disponibles
        demandes = [p for p in cfg["fournisseurs"] if p in disponibles]
        if "CPUExecutionProvider" not in demandes:
            demandes.append("CPUExecutionProvider")
        resultat["demandes"] = demandes

        session = ort.InferenceSession(cfg["modele"], sess_options=options,
                                       providers=demandes)
        alimentation = {}
        for entree in session.get_inputs():
            forme = []
            for index, dimension in enumerate(entree.shape or []):
                if isinstance(dimension, int) and dimension > 0:
                    forme.append(dimension)
                elif index == 0:
                    forme.append(1)
                else:
                    forme.append(int(cfg.get("taille_entree") or 64))
            if not forme:
                forme = [1]
            alimentation[entree.name] = np.zeros(forme, dtype=np.float32)
        session.run(None, alimentation)
        resultat["fournisseur"] = (session.get_providers() or [""])[0]
        resultat["ok"] = True
    except Exception as e:
        resultat["detail"] = str(e)[:600]
    try:
        import onnxruntime as ort
        resultat["disponibles"] = list(ort.get_available_providers())
        resultat["version"] = str(getattr(ort, "__version__", ""))
    except Exception:
        pass
    try:
        with open(sortie, "w", encoding="utf-8") as f:
            json.dump(resultat, f, indent=2)
    except Exception:
        pass
    return 0 if resultat["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def valider_acceleration(python_venv, env, modele, taille_entree, dossier_travail,
                         fournisseurs, progression):
    """Retourne (ok, fournisseur_reel, detail). La decision repose sur le code
    de retour et sur le fichier temoin, jamais sur la presence d'un texte dans
    la sortie du processus."""
    script = os.path.join(dossier_travail, "valider_gpu.py")
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(VALIDATION_SCRIPT)
    temoin = os.path.join(dossier_travail, "validation_gpu.json")
    cfg = os.path.join(dossier_travail, "cfg_validation.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"modele": modele, "fournisseurs": fournisseurs,
                   "taille_entree": taille_entree, "sortie": temoin}, f, indent=2)

    progression("Validation de l'acceleration materielle...")
    log = os.path.join(dossier_travail, "validation_gpu.log")
    with open(log, "w", encoding="utf-8") as lf:
        proc = demarrer_processus([python_venv, script, cfg], env, lf)
        code, depasse, annule = attendre_processus(
            proc, DELAI_VALIDATION_GPU_S, "Validation de l'acceleration materielle...")
    donnees = {}
    try:
        with open(temoin, "r", encoding="utf-8", errors="replace") as f:
            donnees = json.load(f)
    except Exception:
        donnees = {}

    constat = {"ok": False, "fournisseur": donnees.get("fournisseur", ""),
               "detail": "", "disponibles": donnees.get("disponibles") or [],
               "version": donnees.get("version", ""),
               "journal": lignes_diagnostic_moteur(lire_journal(log))}
    if depasse:
        constat["detail"] = ("l'inference de validation a depasse %d s"
                             % DELAI_VALIDATION_GPU_S)
        return constat
    if annule:
        constat["detail"] = "validation interrompue"
        return constat
    if code != 0 or not donnees.get("ok"):
        constat["detail"] = (donnees.get("detail")
                             or lire_journal(log).strip()[-300:]
                             or "l'inference de validation a echoue (code %s)" % code)
        return constat
    constat["ok"] = True
    return constat


def lignes_diagnostic_moteur(journal_texte):
    """Ce que le moteur a dit lui-meme de son chargement de fournisseur.

    onnxruntime ecrit sur la sortie d'erreur la raison exacte pour laquelle il
    n'a pas pu charger un fournisseur - bibliotheque absente, version de cuDNN
    inattendue. C'est la seule source qui nomme la cause ; le greffon, lui, ne
    voit que le symptome.
    """
    interessantes = []
    for ligne in (journal_texte or "").splitlines():
        bas = ligne.lower()
        if any(mot in bas for mot in ("cuda", "cudnn", "provider", "libonnxruntime",
                                      "tensorrt", "rocm", "directml")):
            ligne = ligne.strip()
            if ligne and ligne not in interessantes:
                interessantes.append(ligne[:220])
    return interessantes[-4:]


def installer_cudnn(python_venv, env, dossier_travail, progression):
    """Tente les roues pip de cuDNN, sans jamais en faire un echec bloquant.

    Une dependance systeme que l'utilisateur ne peut pas installer lui-meme
    sans quitter le greffon existe souvent sous forme de paquet pip publie par
    le meme editeur. Le repli absorbe le cas ou aucune ne convient.
    """
    log = os.path.join(dossier_travail, "pip_cudnn.log")
    for paquet in PAQUETS_CUDNN:
        progression("Installation de %s..." % paquet)
        with open(log, "a", encoding="utf-8") as lf:
            proc = demarrer_processus(
                [python_venv, "-m", "pip", "install", "--upgrade",
                 "--only-binary=:all:", paquet], env, lf)
            code, depasse, annule = attendre_processus(
                proc, DELAI_PIP_S, "Installation de %s..." % paquet)
        if annule:
            return False, "installation interrompue"
        if not depasse and code == 0:
            journal("cuDNN installe par pip: " + paquet)
            return True, paquet
    journal("aucune roue cuDNN installable parmi " + ", ".join(PAQUETS_CUDNN))
    return False, ""


def _installer_paquets(py, env, paquets, dossier_travail, progression, quoi):
    """pip en deux passes : roues seules d'abord, puis sans la contrainte.

    Toute commande d'installation releve de la meme regle que le worker : la
    sortie va dans un fichier, jamais dans un PIPE. Un pip install d'une pile
    scientifique produit plusieurs centaines de kilooctets la ou le tampon
    POSIX est de 64 Ko, et le blocage qui s'ensuit se presente comme un
    probleme de reseau.
    """
    log_pip = os.path.join(dossier_travail, "pip_install.log")
    commandes = [
        [py, "-m", "pip", "install", "--upgrade", "--only-binary=:all:"] + paquets,
        [py, "-m", "pip", "install", "--upgrade"] + paquets,
    ]
    dernier_code = None
    for cmd in commandes:
        with open(log_pip, "a", encoding="utf-8") as lf:
            proc = demarrer_processus(cmd, env, lf)
            dernier_code, depasse, annule = attendre_processus(
                proc, DELAI_PIP_S, "Installation de %s..." % quoi)
        if annule:
            raise RuntimeError("RAISON: installation interrompue a la demande de "
                               "l'utilisateur.")
        if depasse:
            invalider_marqueur("timeout pip")
            raise RuntimeError(
                "RAISON: l'installation des dependances a depasse le delai de "
                "%d s et a ete interrompue.\nJournal : %s" % (DELAI_PIP_S, log_pip))
        if dernier_code == 0:
            break
    return dernier_code, log_pip


def preparer_environnement(pile, dossier_travail, reinstaller, progression):
    """Retourne le chemin du Python du venv de la pile demandee, en installant
    ou en reparant au besoin.

    La reparation s'execute a chaque lancement, en dehors de la creation :
    toute la logique de choix de roue placee dans la seule fonction de
    creation ne s'executerait jamais sur une installation deja en place.
    """
    definir_pile(pile)
    if flatpak_detecte():
        raise RuntimeError(
            "GIMP fonctionne dans un bac a sable Flatpak : aucun Python systeme\n"
            "n'y est accessible, et le greffon ne peut donc pas construire son\n"
            "environnement IA. Utilisez une installation de GIMP non Flatpak\n"
            "(paquet systeme, AppImage ou installeur officiel).")

    env = clean_env()
    venv_dir = migrer_anciens_noms(pile)
    py = chemin_python_venv(venv_dir)
    paquets = paquets_de_pile(pile)

    marqueur = lire_marqueur()
    if reinstaller:
        journal("reinstallation demandee par l'utilisateur (pile %s)" % pile)
        marqueur = {}
    elif marqueur_utilisable(marqueur):
        return marqueur["venv_python"]

    # Un venv deja en place et fonctionnel ne se reconstruit pas. C'est aussi
    # ce qui rend le partage avec les autres greffons de la meme pile sans
    # danger : le premier qui arrive installe, les suivants constatent.
    if not reinstaller and os.path.isfile(py):
        progression("Verification de l'environnement IA...")
        if valider_venv(py, env, pile):
            ecrire_marqueur({"statut": "pret", "venv_python": py,
                             "canal": marqueur.get("canal", "existant"),
                             "decouvert_le": marqueur.get(
                                 "decouvert_le", time.strftime("%Y-%m-%dT%H:%M:%S"))})
            return py

    candidats = decouvrir_pythons()
    if not candidats:
        raise RuntimeError(
            "Aucun interpreteur Python 3 utilisable n'a ete trouve sur ce poste\n"
            "(version minimale requise : %s).\n"
            "Installez Python depuis python.org en cochant \"Add python.exe to\n"
            "PATH\", puis relancez le filtre : le greffon fera le reste." %
            ".".join(str(x) for x in PY_MIN))
    choisi = candidats[0]

    if reinstaller and os.path.isdir(venv_dir):
        progression("Suppression de l'environnement IA existant...")
        shutil.rmtree(venv_dir, ignore_errors=True)

    py = chemin_python_venv(venv_dir)
    if not os.path.isfile(py):
        requis = (DISQUE_REQUIS_VENV_GPU_BYTES if pile == STACK_GPU
                  else DISQUE_REQUIS_VENV_BYTES)
        verifier_espace(get_data_dir(), requis, "l'environnement IA (%s)" % pile)
        progression("Creation de l'environnement IA...")
        log_creation = os.path.join(dossier_travail, "creation_venv.log")
        with open(log_creation, "w", encoding="utf-8") as lf:
            proc = demarrer_processus([choisi["chemin"], "-m", "venv", venv_dir], env, lf)
            code, depasse, annule = attendre_processus(
                proc, DELAI_CREATION_VENV_S, "Creation de l'environnement IA...")
        detail = lire_journal(log_creation).strip()
        if annule:
            raise RuntimeError("RAISON: creation interrompue a la demande de "
                               "l'utilisateur.")
        if depasse:
            raise RuntimeError(
                "RAISON: la creation de l'environnement virtuel a depasse le delai "
                "de %d s et a ete interrompue.\nCible : %s"
                % (DELAI_CREATION_VENV_S, venv_dir))
        if code != 0 or not os.path.isfile(py):
            raise RuntimeError(
                "RAISON: echec de la creation de l'environnement virtuel "
                "(code %s).\nCible : %s\nInterpreteur : %s (%s)\nDetails :\n%s"
                % (code, venv_dir, choisi["chemin"], choisi["canal"], detail[-400:]))

    progression("Installation des dependances IA (quelques minutes)...")
    dernier_code, log_pip = _installer_paquets(
        py, env, paquets, dossier_travail, progression, "la pile " + pile)

    if dernier_code != 0 or not valider_venv(py, env, pile):
        detail = lire_journal(log_pip).strip()
        invalider_marqueur("pip code %s" % dernier_code)
        raise RuntimeError(
            "RAISON: les dependances IA n'ont pas pu etre installees (code %s).\n"
            "Interpreteur utilise : %s (%s)\n"
            "Relancez le filtre en cochant \"Reinstaller l'environnement IA\" "
            "apres avoir verifie votre connexion.\nDetails :\n%s"
            % (dernier_code, choisi["chemin"], choisi["canal"], detail[-400:]))

    ecrire_marqueur({
        "statut": "pret",
        "venv_python": py,
        "interpreteur_systeme": choisi["chemin"],
        "canal": choisi["canal"],
        "version_python": ".".join(str(x) for x in choisi["version"]),
        "decouvert_le": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    try:
        with open(get_interpreter_cache(), "w", encoding="utf-8") as f:
            json.dump({"venv_python": py, "canal": choisi["canal"],
                       "decouvert_le": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    except Exception:
        pass
    return py


def verdict_acceleration():
    """Verdict d'acceleration deja constate pour cette pile, ou None.

    La section 17 du scenario est explicite : un echec d'environnement se
    memorise, faute de quoi le greffon relance a chaque ouverture du filtre un
    travail dont il connait deja l'issue. C'etait le cas ici - les roues cuDNN
    etaient retentees et l'inference de controle rejouee a chaque lancement,
    pour reconstater le meme repli.

    Le verdict est lie a la signature des paquets : si la pile change, il
    cesse de valoir.
    """
    verdict = lire_marqueur().get("acceleration")
    if not isinstance(verdict, dict):
        return None
    if verdict.get("signature") != signature_paquets(STACK_GPU):
        return None
    if verdict.get("version_greffon") != PLUGIN_VERSION:
        # Une nouvelle version peut avoir corrige la validation elle-meme :
        # c'est arrive. Un verdict rendu par l'ancienne ne vaut plus.
        return None
    return verdict


def memoriser_verdict_acceleration(ok, constat):
    marqueur = lire_marqueur()
    marqueur["acceleration"] = {
        "verdict": "ok" if ok else "echec",
        "signature": signature_paquets(STACK_GPU),
        "version_greffon": PLUGIN_VERSION,
        "fournisseur": constat.get("fournisseur", ""),
        "detail": str(constat.get("detail", ""))[:300],
        "disponibles": constat.get("disponibles") or [],
        "version_moteur": constat.get("version", ""),
        "journal": constat.get("journal") or [],
        "constate_le": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ecrire_marqueur(marqueur)
    journal("verdict d'acceleration memorise: %s (%s)"
            % (marqueur["acceleration"]["verdict"],
               marqueur["acceleration"]["fournisseur"] or "aucun"))


def taille_dossier(chemin, plafond_fichiers=200000):
    """Taille d'un dossier, bornee en nombre de fichiers parcourus."""
    total = 0
    vus = 0
    for racine, _, fichiers in os.walk(chemin):
        for nom in fichiers:
            vus += 1
            if vus > plafond_fichiers:
                return total
            try:
                total += os.path.getsize(os.path.join(racine, nom))
            except OSError:
                pass
    return total


# Indices lus dans le journal du moteur, du plus precis au plus general. Une
# cause s'etablit sur une trace, elle ne s'affirme pas : la premiere version de
# ce message accusait cuDNN de confiance, alors que le journal disait autre
# chose - une liste de fournisseurs rejetee en bloc, et c'etait un defaut du
# greffon.
INDICES_CAUSE = (
    (("unknown provider type", "ep error"),
     "Le moteur a rejete la liste de fournisseurs demandee, et non un "
     "fournisseur en particulier. C'est un defaut du greffon, pas de ce "
     "poste : signalez ce message."),
    (("cudnn",),
     "cuDNN est en cause : absent, introuvable sur le PATH, ou d'une version "
     "majeure differente de celle qu'attend cette version d'onnxruntime-gpu."),
    (("failed to load library", "loadlibrary", "cannot open shared object"),
     "Une bibliotheque du fournisseur GPU n'a pas pu etre chargee. La ligne "
     "ci-dessus la nomme."),
    (("cuda_error", "cuda failure", "no kernel image"),
     "Le pilote ou le runtime CUDA a refuse l'initialisation."),
)


def cause_probable(constat):
    """Une phrase de cause, appuyee sur ce que le moteur a reellement dit."""
    texte = " ".join(constat.get("journal") or []).lower()
    texte += " " + str(constat.get("detail", "")).lower()
    for motifs, phrase in INDICES_CAUSE:
        if any(motif in texte for motif in motifs):
            return phrase
    disponibles = constat.get("disponibles") or []
    if "CUDAExecutionProvider" not in disponibles:
        return ("Le fournisseur GPU n'est meme pas annonce par le moteur : la "
                "roue installee est une roue processeur, ou sa bibliotheque "
                "n'a pas pu etre chargee.")
    return ("Le fournisseur GPU est annonce disponible mais n'a pas ete "
            "retenu, et le moteur n'en dit pas la raison. Le journal archive "
            "en contient davantage.")


def message_acceleration_ecartee(constat, memorise=False):
    """Le symptome, l'etat des deux couches, la cause probable, et la suite.

    "le moteur est retombe sur CPUExecutionProvider" est un constat, pas un
    diagnostic : l'utilisateur ne peut rien en faire. Le message doit joindre
    ce que declare le moteur et ce que voit le systeme - les deux couches de
    la section 18 - puis dire ce qui se passe maintenant.
    """
    present, detail_runtime = sonde_runtime_cuda()
    disponibles = constat.get("disponibles") or []
    lignes = ["Acceleration materielle ecartee apres une inference de controle."]
    if memorise:
        lignes[0] = ("Acceleration materielle ecartee : constat deja etabli "
                     "lors d'un lancement precedent.")
    lignes.append("  Constat : le calcul s'est fait sur %s."
                  % (constat.get("fournisseur") or "le processeur"))
    lignes.append("  Fournisseurs declares par le moteur : %s"
                  % (", ".join(disponibles) or "aucun constate"))
    lignes.append("  Runtime du systeme : %s" % detail_runtime)
    for ligne in (constat.get("journal") or []):
        lignes.append("  Moteur : " + ligne)
    if constat.get("detail"):
        lignes.append("  Detail : " + premiere_phrase(str(constat["detail"])))
    lignes.append(cause_probable(constat))
    lignes.append("Le traitement se poursuit sur le processeur : meme "
                  "resultat, seulement plus lent.")
    dossier = os.path.join(get_data_dir(), nom_venv(STACK_GPU))
    if os.path.isdir(dossier):
        lignes.append("L'environnement GPU installe occupe %s ici :"
                      % octets_lisibles(taille_dossier(dossier)))
        lignes.append("  " + dossier)
    lignes.append("Ce constat est memorise : les lancements suivants passeront "
                  "directement au processeur, sans rien reinstaller ni "
                  "reessayer. Pour refaire l'essai, cochez \"Reinstaller "
                  "l'environnement IA\".")
    return "\n".join(lignes)


def refus_gpu(detail):
    """Message du seul endroit ou le greffon decline une demande explicite de
    l'utilisateur : il doit donc expliquer ce qui manque et comment s'en
    passer, et ne transferer aucun octet."""
    return (
        "L'option \"Utiliser la carte graphique\" a ete refusee avant tout "
        "telechargement : ce poste n'expose pas le runtime CUDA dont la pile "
        "onnxruntime-gpu a besoin.\n"
        "Constat : %s\n"
        "onnxruntime-gpu n'embarque pas son runtime, contrairement aux roues "
        "de PyTorch : un pilote graphique ne suffit pas, il faut le CUDA "
        "Toolkit.\n"
        "Le traitement se poursuit sur le processeur, ce qui donne le meme "
        "resultat, seulement plus lentement." % detail)


# ==============================================================================
# 6. MODELES : RECHERCHE, TELECHARGEMENT ANNONCE, DEPOT MANUEL, DEGRADATION
#    Diagnostiquer une absence n'est pas la reparer. Le greffon sait construire
#    le chemin, verifier la taille, calculer une empreinte : il doit aussi
#    savoir aller chercher le fichier, et se passer de lui quand il n'y arrive
#    pas.
# ==============================================================================
def noms_de_modeles_connus():
    return set(entree["fichier"] for entree in MODELES.values())


def chercher_modele(nom_fichier):
    """Cherche le fichier recursivement sous le dossier de modeles gere par la
    suite, puis dans les emplacements de repli.

    L'emplacement des poids appartient a la bibliotheque, pas au greffon : une
    recherche sur un seul chemin repondrait "absent" en permanence des que la
    disposition change d'une version a l'autre.
    """
    racine = get_models_dir()
    direct = os.path.join(racine, nom_fichier)
    if os.path.isfile(direct):
        return direct, "canonique"
    if os.path.isdir(racine):
        for dossier, _, fichiers in os.walk(racine):
            if nom_fichier in fichiers:
                return os.path.join(dossier, nom_fichier), "sous_dossier_gere"
    ancien = os.path.join(get_shared_dir(), "models", nom_fichier)
    if os.path.isfile(ancien):
        return ancien, "ancien_emplacement"
    # Repli de depannage : a cote du greffon, dans plug-ins/, l'arborescence
    # que GIMP parcourt a chaque demarrage. Un modele pose la echappe a la
    # mutualisation et sera retelecharge par chaque greffon de la suite.
    try:
        local = os.path.join(os.path.dirname(os.path.realpath(__file__)), nom_fichier)
        if os.path.isfile(local):
            return local, "a_cote_du_greffon"
    except Exception:
        pass
    return None, None


def fichier_semble_onnx(chemin):
    """Rejette ce qui n'est manifestement pas un modele : page HTML d'erreur,
    JSON, archive, image, pointeur Git LFS. Ne pretend pas valider le graphe."""
    try:
        with open(chemin, "rb") as f:
            tete = f.read(65536)
    except Exception as e:
        return False, "lecture impossible (%s)" % e
    if not tete:
        return False, "fichier vide"
    debuts_refuses = [
        (b"<", "document HTML ou XML"),
        (b"{", "document JSON"),
        (b"PK\x03\x04", "archive ZIP"),
        (b"\x89PNG", "image PNG"),
        (b"version https://git-lfs", "pointeur Git LFS, pas le fichier reel"),
    ]
    for prefixe, quoi in debuts_refuses:
        if tete.startswith(prefixe):
            return False, quoi
    return True, "protobuf ONNX plausible"


def taille_min_modele(nom_fichier):
    """Plancher de vraisemblance, par modele.

    Un plancher unique pour toute la table serait une hypothese sur la taille
    des modeles : YuNet pese 227 Ko la ou les autres pesent des centaines de
    megaoctets, et un plancher a un megaoctet le rejetterait purement et
    simplement.
    """
    for entree in MODELES.values():
        if entree["fichier"] == nom_fichier:
            return int(entree.get("taille_min") or MODELE_TAILLE_MIN_BYTES)
    return MODELE_TAILLE_MIN_BYTES


def controler_vraisemblance(chemin, nom_fichier):
    """Controle de taille en amont du hash : un fichier nul, tronque ou
    aberrant est rejete avant meme d'etre lu integralement."""
    taille = os.path.getsize(chemin)
    plancher = taille_min_modele(nom_fichier)
    if taille < plancher:
        raise ValueError("Le fichier '%s' est tronque (%s, minimum attendu %s)."
                         % (nom_fichier, octets_lisibles(taille),
                            octets_lisibles(plancher)))
    if taille > MODELE_TAILLE_MAX_BYTES:
        raise ValueError("Le fichier '%s' depasse la taille vraisemblable d'un "
                         "modele de cette suite (%s)."
                         % (nom_fichier, octets_lisibles(taille)))
    ok, detail = fichier_semble_onnx(chemin)
    if not ok:
        raise ValueError("Le fichier '%s' n'est pas un modele ONNX : %s."
                         % (nom_fichier, detail))
    return taille


def empreinte_fichier(chemin):
    """SHA-256 avec cache sur (taille, mtime).

    A un gigaoctet, rehacher a chaque lancement n'est pas une optimisation de
    confort mais une condition de fonctionnement. Le cache ne detecte pas un
    remplacement qui preserve ces deux valeurs : il vise la corruption
    accidentelle, pas la malveillance.
    """
    cache_path = os.path.join(get_shared_dir(), "empreintes_cache.json")
    try:
        stat = os.stat(chemin)
        cle = os.path.normpath(chemin).lower()
        signature = [stat.st_size, int(stat.st_mtime)]
    except Exception:
        return ""
    cache = {}
    try:
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            cache = json.load(f)
    except Exception:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    entree = cache.get(cle)
    if isinstance(entree, dict) and entree.get("signature") == signature:
        return entree.get("sha256", "")
    h = hashlib.sha256()
    try:
        with open(chemin, "rb") as f:
            for bloc in iter(lambda: f.read(1024 * 1024), b""):
                h.update(bloc)
    except Exception:
        return ""
    digest = h.hexdigest()
    cache[cle] = {"signature": signature, "sha256": digest}
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass
    return digest


def tofu(nom_fichier, digest, avertissements):
    """Confiance a la premiere utilisation : memorise l'empreinte au premier
    usage, compare ensuite, avertit sans jamais bloquer.

    Le greffon n'exerce aucun controle d'empreinte de reference et n'en promet
    aucun : la provenance des poids varie selon l'utilisateur, un hash fige
    bloquerait tout fichier legitime issu d'une autre source. Le TOFU detecte
    un changement, jamais une malveillance.
    """
    if not digest:
        return
    path = get_tofu_path()
    connues = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            connues = json.load(f)
    except Exception:
        connues = {}
    if not isinstance(connues, dict):
        connues = {}
    if nom_fichier in connues:
        if connues[nom_fichier] != digest:
            avertissements.append(
                "L'empreinte du modele '%s' a change depuis sa premiere "
                "utilisation. Traitement poursuivi." % nom_fichier)
            connues[nom_fichier] = digest
        else:
            return
    else:
        connues[nom_fichier] = digest
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(connues, f, indent=2)
    except Exception:
        pass


def motif_reseau(erreur):
    """Traduit un echec reseau en une phrase qui oriente l'utilisateur.

    "HTTP Error 401: Unauthorized" ne dit rien a qui veut simplement un
    modele. Un 401 ou un 403 sur un depot public signifie, neuf fois sur dix,
    que le depot a disparu ou est devenu prive - pas que le reseau est coupe,
    et donc pas qu'il faut reessayer plus tard.
    """
    code = getattr(erreur, "code", None)
    if code in (401, 403):
        return ("cette adresse demande une authentification ou n'existe plus "
                "(HTTP %s)" % code)
    if code == 404:
        return "cette adresse n'existe plus (HTTP 404)"
    if code == 429:
        return "le serveur refuse temporairement les telechargements (HTTP 429)"
    if isinstance(code, int) and code >= 500:
        return "le serveur est en panne (HTTP %s)" % code
    return premiere_phrase(str(erreur))


def taille_distante(url):
    """Taille annoncee par le serveur, par une requete HEAD. Retourne -1 si
    elle n'est pas connue : aucun octet du corps n'a ete transfere."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "gimp-ai-suite/" + PLUGIN_VERSION)
        with urllib.request.urlopen(req, timeout=DELAI_LECTURE_RESEAU_S) as rep:
            valeur = rep.headers.get("Content-Length")
            return int(valeur) if valeur else -1
    except Exception as e:
        journal("HEAD impossible sur %s : %s" % (url, e))
        return -1


def _telecharger_url(url, nom_fichier, reference, progression):
    import urllib.request

    destination = os.path.join(get_models_dir(), nom_fichier)
    partiel = destination + ".part"
    limite = time.time() + DELAI_TELECHARGEMENT_S
    recus = 0
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "gimp-ai-suite/" + PLUGIN_VERSION)
        with urllib.request.urlopen(req, timeout=DELAI_LECTURE_RESEAU_S) as rep:
            entete = rep.headers.get("Content-Length")
            total = int(entete) if entete else reference
            if total > AUTO_DOWNLOAD_MAX_BYTES:
                raise RuntimeError(
                    "Le modele '%s' pese %s, au-dela du seuil de telechargement "
                    "automatique (%s)." % (nom_fichier, octets_lisibles(total),
                                           octets_lisibles(AUTO_DOWNLOAD_MAX_BYTES)))
            with open(partiel, "wb") as f:
                while True:
                    if time.time() > limite:
                        raise RuntimeError(
                            "Telechargement de '%s' interrompu apres %d s (%s recus)."
                            % (nom_fichier, DELAI_TELECHARGEMENT_S,
                               octets_lisibles(recus)))
                    bloc = rep.read(1024 * 256)
                    if not bloc:
                        break
                    f.write(bloc)
                    recus += len(bloc)
                    if total > 0:
                        fraction = min(1.0, max(0.0, float(recus) / float(total)))
                        try:
                            Gimp.progress_update(fraction)
                        except Exception:
                            pass
                        progression("Telechargement de %s : %s sur %s"
                                    % (nom_fichier, octets_lisibles(recus),
                                       octets_lisibles(total)))
                    else:
                        try:
                            Gimp.progress_pulse()
                        except Exception:
                            pass
        controler_vraisemblance(partiel, nom_fichier)
        os.replace(partiel, destination)
    except Exception:
        try:
            if os.path.isfile(partiel):
                os.remove(partiel)
        except Exception:
            pass
        raise
    journal("telechargement termine: %s (%s)" % (destination, octets_lisibles(recus)))
    return destination, recus


def telecharger_modele(role, progression):
    """Telecharge un modele apres avoir annonce sa taille reelle.

    Les adresses sont essayees dans l'ordre : une adresse morte ne condamne pas
    le modele tant qu'une autre repond. Aucune version d'un depot n'est figee
    en dur ; la liste se complete par sources_modeles.json.
    """
    entree = MODELES[role]
    nom_fichier = entree["fichier"]
    sources = sources_du_modele(role)
    if not sources:
        raise RuntimeError(
            "RAISON: aucune adresse de telechargement n'est declaree pour le "
            "modele '%s'." % nom_fichier)

    declaree = entree.get("taille_declaree", 0)
    echecs = []
    for url in sources:
        taille = taille_distante(url)
        reference = taille if taille > 0 else declaree
        if reference > AUTO_DOWNLOAD_MAX_BYTES:
            echecs.append("%s : %s annonces, au-dela du seuil de %s"
                          % (url, octets_lisibles(reference),
                             octets_lisibles(AUTO_DOWNLOAD_MAX_BYTES)))
            continue
        requis = reference if reference > 0 else MODELE_TAILLE_MIN_BYTES
        verifier_espace(get_models_dir(), requis, "le modele '%s'" % nom_fichier)
        if reference > 0:
            progression("Telechargement de %s (%s)..."
                        % (nom_fichier, octets_lisibles(reference)))
        else:
            progression("Telechargement de %s (taille inconnue)..." % nom_fichier)
        journal("telechargement %s (%s annonces)" % (url, octets_lisibles(reference)))
        try:
            return _telecharger_url(url, nom_fichier, reference, progression)
        except Exception as e:
            echecs.append("%s : %s" % (url, motif_reseau(e)))
            journal("source en echec %s : %s" % (url, e))
    raise RuntimeError("RAISON: aucune source n'a fourni '%s' (%s)."
                       % (nom_fichier, " | ".join(echecs)[:400]))


def ranger_modele(chemin, nom_fichier):
    """Deplace un modele trouve dans un sous-dossier gere vers l'emplacement
    canonique, puis supprime les copies devenues inertes.

    Un greffon qui cree un dossier doit savoir y faire le menage, sinon il
    laisse plusieurs centaines de megaoctets immobilises que personne ne
    remarquera. La suppression n'est legitime qu'a trois conditions : le
    fichier est dans un dossier que le greffon gere lui-meme, il porte le nom
    d'un modele connu, et sa taille est identique a celle de l'exemplaire
    conserve. Une taille differente designe un fichier qu'on ne reconnait
    pas : on n'y touche pas.
    """
    canonique = os.path.join(get_models_dir(), nom_fichier)
    if nom_fichier not in noms_de_modeles_connus():
        return chemin
    geres = (get_models_dir(), os.path.join(get_shared_dir(), "models"))
    try:
        if os.path.normpath(chemin) == os.path.normpath(canonique):
            pass
        elif any(chemin.startswith(racine) for racine in geres):
            os.replace(chemin, canonique)
            journal("modele range: %s -> %s" % (chemin, canonique))
            chemin = canonique
        else:
            return chemin
    except Exception as e:
        journal("rangement impossible (%s)" % e)
        return chemin

    try:
        reference = os.path.getsize(canonique)
    except Exception:
        return chemin
    for racine in geres:
        if not os.path.isdir(racine):
            continue
        for dossier, _, fichiers in os.walk(racine):
            if nom_fichier not in fichiers:
                continue
            doublon = os.path.join(dossier, nom_fichier)
            if os.path.normpath(doublon) == os.path.normpath(canonique):
                continue
            try:
                if os.path.getsize(doublon) == reference:
                    os.remove(doublon)
                    journal("doublon inerte supprime: " + doublon)
            except Exception:
                pass
    return chemin


def ecarter_fichier_inutilisable(chemin, nom_fichier, raison, avertissements):
    """Supprime un fichier de modele provablement inutilisable, mais seulement
    dans un dossier que le greffon gere lui-meme et sous un nom de modele
    connu.

    C'est le cas d'un telechargement interrompu : il porte le bon nom, le
    greffon le trouve a chaque lancement, le refuse au controle de
    vraisemblance et echoue indefiniment puisque rien ne le remplace.
    """
    gere = (chemin.startswith(get_models_dir())
            or chemin.startswith(os.path.join(get_shared_dir(), "models")))
    if not gere:
        return False
    try:
        os.remove(chemin)
    except Exception:
        return False
    journal("modele inutilisable supprime (%s): %s" % (raison, chemin))
    avertissements.append(
        "Le fichier '%s' etait inutilisable (%s) et a ete supprime pour "
        "permettre un nouveau telechargement." % (nom_fichier, raison))
    return True


def obtenir_modele(role, autoriser_telechargement, progression, avertissements):
    """Retourne le chemin d'un modele utilisable, en le telechargeant si
    besoin. Leve RuntimeError si le fichier reste indisponible."""
    nom_fichier = MODELES[role]["fichier"]
    chemin, origine = chercher_modele(nom_fichier)
    if chemin:
        chemin = ranger_modele(chemin, nom_fichier)
        try:
            controler_vraisemblance(chemin, nom_fichier)
        except ValueError as e:
            if not ecarter_fichier_inutilisable(chemin, nom_fichier,
                                                premiere_phrase(str(e)),
                                                avertissements):
                raise
            chemin = None
        if chemin and origine == "a_cote_du_greffon":
            avertissements.append(
                "Le modele '%s' est utilise depuis le dossier du greffon. "
                "Deplacez-le dans %s pour qu'il serve a toute la suite."
                % (nom_fichier, get_models_dir()))
        if chemin:
            tofu(nom_fichier, empreinte_fichier(chemin), avertissements)
            return chemin

    if not autoriser_telechargement:
        raise RuntimeError(
            "RAISON: le modele '%s' est absent et le telechargement automatique "
            "est decoche." % nom_fichier)

    chemin, _ = telecharger_modele(role, progression)
    controler_vraisemblance(chemin, nom_fichier)
    tofu(nom_fichier, empreinte_fichier(chemin), avertissements)
    return chemin


def message_depot_manuel(roles, entete=None):
    """Message de dernier recours : chemin de depot simple, collable tel quel,
    et nom exact du fichier attendu.

    Il n'apparait que si aucune variante n'est disponible - typiquement un
    poste sans reseau. Le chemin annonce est celui que le greffon lit
    reellement, sinon le fichier serait depose correctement et retelecharge
    quand meme.
    """
    lignes = [entete or "Aucun modele utilisable pour cette operation."]
    lignes.append("")
    lignes.append("Deposez le ou les fichiers suivants dans ce dossier, puis "
                  "relancez le filtre :")
    lignes.append("  " + get_models_dir())
    lignes.append("")
    for role in roles:
        entree = MODELES[role]
        lignes.append("  %s  (%s, entree %dx%d)"
                      % (entree["fichier"],
                         octets_lisibles(entree.get("taille_declaree", 0)),
                         entree["taille_entree"], entree["taille_entree"]))
        for url in sources_du_modele(role):
            lignes.append("      " + url)
    lignes.append("")
    lignes.append("Un autre export ONNX du meme modele convient s'il respecte "
                  "ce format d'entree.")
    lignes.append("")
    lignes.append("Une adresse de telechargement peut aussi etre declaree dans "
                  "ce fichier :")
    lignes.append("  " + get_sources_utilisateur_path())
    lignes.append("  Exemple : {\"%s\": [\"https://exemple.org/%s\"]}"
                  % (roles[0], MODELES[roles[0]]["fichier"]))
    return "\n".join(lignes)


# Roles qui servent la meme capacite. Si l'un repond, l'absence de l'autre
# n'est pas une degradation et ne merite aucun message : deux detecteurs pour
# une seule detection.
ROLES_ALTERNATIFS = ((ROLE_DETECTION, ROLE_DETECTION_YUNET),)


def resoudre_modeles(roles, autoriser_telechargement, progression, avertissements):
    """Resout les modeles des roles demandes, sans jamais interrompre.

    Un modele indisponible degrade : l'operation passe sur son chemin sans IA
    quand il en existe un, ou elle est annulee et le motif est dit en une
    phrase.

    Et le message dit comment y remedier. Diagnostiquer une absence n'est pas
    la reparer : constater qu'un modele manque sans indiquer ni ou le deposer
    ni comment en declarer l'adresse oblige l'utilisateur a quitter GIMP pour
    chercher, ce qui est exactement ce que ce greffon ne doit jamais demander.
    """
    resolus = {}
    motifs = {}
    for role in roles:
        progression("Verification du modele %s..." % MODELES[role]["fichier"])
        try:
            resolus[role] = obtenir_modele(role, autoriser_telechargement,
                                           progression, avertissements)
        except Exception as e:
            resolus[role] = None
            motifs[role] = premiere_phrase(str(e))
            journal("modele %s indisponible : %s" % (role, motifs[role]))

    couverts = set()
    for groupe in ROLES_ALTERNATIFS:
        if any(resolus.get(autre) for autre in groupe):
            couverts.update(groupe)

    manquants = [role for role in roles
                 if resolus.get(role) is None and role not in couverts]
    for role in manquants:
        if role in ROLES_AVEC_REPLI:
            avertissements.append(
                "Modele '%s' indisponible (%s) : l'etape %s se fait sans IA."
                % (MODELES[role]["fichier"], motifs.get(role, "motif inconnu"),
                   role))
        else:
            avertissements.append(
                "Modele '%s' indisponible (%s) : l'etape %s a ete ignoree."
                % (MODELES[role]["fichier"], motifs.get(role, "motif inconnu"),
                   role))
    if manquants:
        avertissements.append(message_depot_manuel(
            manquants,
            "Voici comment fournir le ou les modeles manquants."))
    return resolus


def ecrire_inventaire(details):
    """Inventaire de ce que le greffon a cree hors de son dossier temporaire.

    Un greffon qui fonctionne peut gaspiller en silence : les doublons de
    modeles d'une serie precedente n'ont ete decouverts ni par le code, ni par
    les tests, ni par les journaux d'incident - il n'y avait pas d'incident.
    """
    inventaire = {
        "greffon": PLUGIN_ID,
        "version_greffon": PLUGIN_VERSION,
        "pile": pile_active(),
        "mis_a_jour_le": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dossier_donnees": get_data_dir(),
        "dossier_partage": get_shared_dir(),
        "venv": os.path.join(get_data_dir(), nom_venv(pile_active())),
        "marqueur": get_marker_path(),
        "modeles": [],
    }
    racine = get_models_dir()
    total = 0
    try:
        for dossier, _, fichiers in os.walk(racine):
            for nom in sorted(fichiers):
                if not nom.endswith(".onnx"):
                    continue
                chemin = os.path.join(dossier, nom)
                try:
                    taille = os.path.getsize(chemin)
                except Exception:
                    continue
                total += taille
                inventaire["modeles"].append(
                    {"nom": nom, "chemin": chemin, "octets": taille,
                     "lisible": octets_lisibles(taille),
                     "connu_de_ce_greffon": nom in noms_de_modeles_connus()})
    except Exception:
        pass
    inventaire["total_modeles_octets"] = total
    inventaire["total_modeles_lisible"] = octets_lisibles(total)
    inventaire.update(details or {})
    try:
        with open(get_inventory_path(), "w", encoding="utf-8") as f:
            json.dump(inventaire, f, indent=2)
    except Exception:
        pass
    return inventaire


# ==============================================================================
# 7. ARCHIVAGE DES JOURNAUX
#    La section 13 impose la suppression du dossier d'execution ; elle ne doit
#    pas emporter la seule trace exploitable. Un utilisateur ne posera jamais
#    une variable d'environnement, ni ne lancera GIMP depuis un terminal, pour
#    produire un rapport de bogue : si le diagnostic depend de ce geste, il
#    n'existe pas.
#
#    Le dossier logs/ est partage par toute la suite. Chaque archive porte un
#    incident.json qui nomme le greffon qui l'a produite, et la purge ne
#    s'applique qu'aux siennes : voir archives_du_greffon().
# ==============================================================================
NOM_FICHIER_INCIDENT = "incident.json"
# Age au-dela duquel une archive que personne ne revendique peut etre
# supprimee. Elles viennent des versions anterieures a ce marqueur, et des
# greffons de la suite qui ne le posent pas encore.
JOURS_ARCHIVES_ORPHELINES = 30


def _inventaire_archives(racine):
    """Repartit les archives de logs/ : les miennes, et celles que personne ne
    revendique. Chacune triee de la plus ancienne a la plus recente, datee par
    son contenu et non par son nom.

    Le dossier logs/ est partage, et deux conventions de nommage y cohabitent
    dans la suite : "2026-09-16_19-44-05" pour certains greffons,
    "20260916-194405-000123" pour d'autres. En ASCII le tiret (0x2D) precede le
    chiffre (0x30) : un tri alphabetique place donc systematiquement la
    premiere forme en tete, et une purge qui s'y fierait supprimerait toujours
    les archives des autres greffons avant les siennes, quel que soit leur age.
    Dix incidents d'un greffon suffiraient a effacer tout l'historique d'un
    autre - sans le moindre message, puisqu'il n'y a pas d'incident quand un
    greffon fonctionne.

    Deux consequences, tirees de ce constat :

    - on ne purge que ce qu'on a produit, reconnaissable a son incident.json ;
    - on date par la date du fichier, pas par le nom du dossier, pour ne
      dependre d'aucune convention de nommage.

    Une archive dont l'incident.json n'a pas pu etre ecrit n'est jamais
    supprimee : elle restera, ce qui est preferable a effacer celle d'un
    voisin.
    """
    miennes = []
    orphelines = []
    try:
        noms = os.listdir(racine)
    except Exception:
        return [], []
    for nom in noms:
        chemin = os.path.join(racine, nom)
        if not os.path.isdir(chemin):
            continue
        marque = os.path.join(chemin, NOM_FICHIER_INCIDENT)
        donnees = None
        if os.path.isfile(marque):
            try:
                with open(marque, "r", encoding="utf-8", errors="replace") as f:
                    donnees = json.load(f)
            except Exception:
                donnees = None
        if isinstance(donnees, dict) and donnees.get("greffon") == PLUGIN_ID:
            try:
                date = os.path.getmtime(marque)
            except Exception:
                date = 0.0
            miennes.append((date, nom, chemin))
        elif not isinstance(donnees, dict):
            # Personne ne revendique cette archive : elle est anterieure au
            # marqueur, ou vient d'un greffon qui ne le pose pas encore.
            try:
                date = os.path.getmtime(chemin)
            except Exception:
                date = 0.0
            orphelines.append((date, nom, chemin))
    miennes.sort()
    orphelines.sort()
    return miennes, orphelines


def archives_du_greffon(racine):
    """Archives produites par ce greffon, de la plus ancienne a la plus
    recente."""
    return [chemin for _, _, chemin in _inventaire_archives(racine)[0]]


def purger_journaux(racine):
    """Purge les archives de ce greffon, et resorbe le stock orphelin.

    Une archive que personne ne revendique n'est supprimee qu'a deux
    conditions reunies : etre plus vieille que JOURS_ARCHIVES_ORPHELINES, et
    ne pas figurer parmi les ARCHIVES_A_CONSERVER plus recentes d'entre elles.
    Un greffon de la suite qui n'aurait pas encore recu ce correctif garde
    ainsi ses archives recentes, et le stock ancien se resorbe quand meme.

    Retourne les chemins supprimes, pour que le comportement soit verifiable
    autrement que par une inspection du dossier.
    """
    miennes, orphelines = _inventaire_archives(racine)
    limite = time.time() - JOURS_ARCHIVES_ORPHELINES * 86400
    condamnees = [chemin for _, _, chemin in miennes[:-ARCHIVES_A_CONSERVER]]
    condamnees += [chemin for date, _, chemin
                   in orphelines[:-ARCHIVES_A_CONSERVER] if date < limite]
    for chemin in condamnees:
        shutil.rmtree(chemin, ignore_errors=True)
    return condamnees


def archiver_journaux(dossier_travail, contexte=None):
    try:
        racine = get_logs_dir()
        # Deux incidents dans la meme seconde ne doivent pas s'ecraser : c'est
        # justement dans une serie d'echecs rapproches que les journaux
        # comptent.
        horodatage = "%s-%06d" % (time.strftime("%Y%m%d-%H%M%S"),
                                  time.time_ns() // 1000 % 1000000)
        cible = os.path.join(racine, horodatage)
        os.makedirs(cible, exist_ok=True)
        # L'incident est marque avant la copie : si celle-ci echoue a
        # mi-chemin, l'archive reste identifiable, donc purgeable.
        try:
            with open(os.path.join(cible, NOM_FICHIER_INCIDENT), "w",
                      encoding="utf-8") as f:
                json.dump({"greffon": PLUGIN_ID, "version": PLUGIN_VERSION,
                           "pile": pile_active(), "api_gimp": API_GIMP,
                           "plateforme": sys.platform,
                           "horodatage": horodatage,
                           "contexte": contexte or {}}, f, indent=2)
        except Exception:
            pass
        for nom in os.listdir(dossier_travail):
            if not (nom.endswith(".log") or nom.endswith(".json")
                    or nom.endswith(".py")):
                continue
            try:
                shutil.copy2(os.path.join(dossier_travail, nom),
                             os.path.join(cible, nom))
            except Exception:
                pass
        purger_journaux(racine)
        return cible
    except Exception as e:
        journal("archivage impossible: %s" % e)
        return None


def premiere_phrase(texte):
    """Un message repris dans un autre ne doit pas trainer sa queue de journal.

    Quand un message est destine a etre cite ailleurs, la raison est delimitee
    par le prefixe repere RAISON: ; a defaut, seule la premiere phrase est
    conservee.
    """
    texte = (texte or "").strip()
    for ligne in texte.splitlines():
        if ligne.startswith("RAISON:"):
            return ligne[len("RAISON:"):].strip()
    if not texte:
        return "cause inconnue"
    premiere = texte.splitlines()[0]
    for separateur in (". ", " : "):
        if separateur in premiere:
            return premiere.split(separateur)[0].strip()
    return premiere.strip()

# ==============================================================================
# 8. COMPATIBILITE DE L'API GIMP 3.0
# ==============================================================================
def lire_option(config, nom, defaut):
    """Lit une propriete en retournant le defaut si elle n'existe pas : un
    add_*_argument qui a echoue ne doit pas faire tomber l'execution."""
    try:
        valeur = config.get_property(nom)
        return defaut if valeur is None else valeur
    except Exception:
        return defaut


def bornes_selection(image):
    """(selection_active, x, y, largeur, hauteur) en coordonnees image.

    Les fonctions de l'API qui renvoient plusieurs valeurs n'ont pas une
    disposition stable : un booleen de succes peut preceder les valeurs utiles
    et decaler tous les indices. On filtre donc les booleens et on prend les
    quatre entiers, sans dependre de leur position exacte.
    """
    largeur, hauteur = image.get_width(), image.get_height()
    try:
        retour = Gimp.Selection.bounds(image)
    except Exception:
        return False, 0, 0, largeur, hauteur
    if not isinstance(retour, (tuple, list)):
        return False, 0, 0, largeur, hauteur

    booleens = [v for v in retour if isinstance(v, bool)]
    entiers = [int(v) for v in retour if isinstance(v, int) and not isinstance(v, bool)]
    active = booleens[0] if booleens else bool(entiers and len(entiers) >= 4)
    if not active or len(entiers) < 4:
        return False, 0, 0, largeur, hauteur

    x1, y1, x2, y2 = entiers[-4:]
    x1 = max(0, min(largeur, x1))
    y1 = max(0, min(hauteur, y1))
    x2 = max(0, min(largeur, x2))
    y2 = max(0, min(hauteur, y2))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False, 0, 0, largeur, hauteur
    return True, x1, y1, w, h


def _tenter_export(image, drawable, chemin):
    fichier = Gio.File.new_for_path(chemin)
    uri = None
    try:
        uri = GLib.filename_to_uri(chemin)
    except Exception:
        uri = None

    tentatives = [
        lambda: Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, [drawable], fichier),
        lambda: Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, fichier),
    ]
    if uri:
        tentatives.append(
            lambda: Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, [drawable], uri))
        tentatives.append(
            lambda: Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, drawable, uri))

    def produit():
        # Une variante peut ne lever aucune exception tout en ne produisant
        # aucun fichier : on verifie le resultat, pas l'absence d'erreur.
        return os.path.exists(chemin) and os.path.getsize(chemin) > 0

    for tentative in tentatives:
        try:
            tentative()
        except Exception:
            continue
        if produit():
            return True

    try:
        proc = Gimp.get_pdb().lookup_procedure("file-png-save")
        if proc:
            cfg = proc.create_config()
            cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
            cfg.set_property("image", image)
            cfg.set_property("file", fichier)
            try:
                cfg.set_property("drawables", [drawable])
            except Exception:
                pass
            proc.run(cfg)
    except Exception:
        pass
    return produit()


def exporter_calque(image, drawable, chemin):
    """Exporte le calque designe, pas le composite visible.

    Gimp.file_save applique a l'image exporte l'aplatissement des calques
    visibles : sur un document a plusieurs calques, le traitement porterait
    donc sur autre chose que ce que l'utilisateur a selectionne. On construit
    une image tampon de la taille du canevas ne contenant que ce calque, a son
    decalage d'origine, pour que les coordonnees restent celles de l'image.
    """
    tampon = None
    try:
        largeur, hauteur = image.get_width(), image.get_height()
        try:
            tampon = Gimp.Image.new_with_precision(
                largeur, hauteur, image.get_base_type(), image.get_precision())
        except Exception:
            tampon = Gimp.Image.new(largeur, hauteur, image.get_base_type())

        copie = Gimp.Layer.new_from_drawable(drawable, tampon)
        tampon.insert_layer(copie, None, 0)
        try:
            decalage = drawable.get_offsets()
            entiers = [int(v) for v in decalage
                       if isinstance(v, int) and not isinstance(v, bool)]
            if len(entiers) >= 2:
                copie.set_offsets(entiers[-2], entiers[-1])
        except Exception:
            pass
        for reglage, valeur in (("set_visible", True), ("set_opacity", 100.0)):
            try:
                getattr(copie, reglage)(valeur)
            except Exception:
                pass
        try:
            copie.set_mode(Gimp.LayerMode.NORMAL)
        except Exception:
            pass
        try:
            if not copie.has_alpha():
                copie.add_alpha()
        except Exception:
            pass

        if _tenter_export(tampon, copie, chemin):
            return True
    except Exception as e:
        journal("export par image tampon impossible: %s" % e)
    finally:
        if tampon is not None:
            try:
                tampon.delete()
            except Exception:
                pass

    # Repli : export direct. Le traitement portera sur le composite visible,
    # ce qui est signale a l'utilisateur par l'appelant.
    if _tenter_export(image, drawable, chemin):
        return False
    raise RuntimeError("RAISON: aucune variante de l'API d'export GIMP n'a "
                       "produit le fichier temporaire attendu.")


def charger_calque(image, chemin):
    """Charge un PNG comme calque de l'image, avec repli complet."""
    fichier = Gio.File.new_for_path(chemin)
    try:
        calque = Gimp.file_load_layer(Gimp.RunMode.NONINTERACTIVE, image, fichier)
        if calque is not None:
            return calque
    except Exception:
        pass
    try:
        calques = Gimp.file_load_layers(Gimp.RunMode.NONINTERACTIVE, image, fichier)
        if calques:
            return calques[0]
    except Exception:
        pass
    # Dernier repli : image temporaire puis calque depuis le visible.
    temporaire = None
    try:
        temporaire = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, fichier)
        calque = Gimp.Layer.new_from_visible(temporaire, image, "resultat")
        return calque
    except Exception as e:
        raise RuntimeError("RAISON: impossible de charger le resultat dans "
                           "l'image (%s)." % e)
    finally:
        if temporaire is not None:
            try:
                temporaire.delete()
            except Exception:
                pass



# ==============================================================================
# 9. SCRIPT WORKER ISOLE
#    Aucune interpolation : tous les parametres arrivent par un fichier JSON
#    dont le chemin est le seul argument. Les chemins exotiques, espaces et
#    accents cessent d'etre un sujet.
#
#    En contrepartie, marqueurs et cles de configuration sont ecrits deux fois.
#    outils/verifier_livraison.py extrait les deux ensembles et les compare :
#    une cle lue ici et jamais ecrite par le greffon ferait travailler le
#    worker avec une autre valeur que celle livree, sans rien signaler, et le
#    symptome serait un reglage qui "ne fait rien".
#
#    Les bibliotheques lourdes ne sont importees qu'au moment precis ou une
#    fonction en a besoin, jamais en tete de fichier : l'empreinte memoire a
#    l'instant T ne depasse ainsi jamais celle du modele le plus lourd de la
#    chaine.
# ==============================================================================
WORKER_SCRIPT = r'''# -*- coding: utf-8 -*-

import gc
import json
import os
import sys
import time
import traceback

CODE_OK = 0
CODE_PARAMS = 2
CODE_IMPORT = 3
CODE_IMAGE = 4
CODE_MODELE = 5
CODE_PIVOT = 6
CODE_INFERENCE = 7
CODE_AUCUN_VISAGE = 8
CODE_ECRITURE = 9
CODE_MEMOIRE = 10
CODE_INATTENDU = 11


class ErreurModele(Exception):
    """Le fichier de modele n'a pas pu etre charge."""


class ErreurInference(Exception):
    """Le modele s'est charge, mais le calcul a echoue."""


# Echecs rencontres par les etapes. Ils sont conserves pour que, si rien n'est
# produit au bout du compte, le marqueur emis designe la cause reelle plutot
# qu'un symptome : un modele absent et une image sans visage ne se reparent pas
# de la meme facon.
ECHECS_MODELES = []


def noter_echec(role, exception):
    if isinstance(exception, ErreurModele):
        genre = "modele"
    elif isinstance(exception, ErreurInference):
        genre = "inference"
    else:
        genre = "etape"
    ECHECS_MODELES.append({"role": role, "genre": genre,
                           "detail": str(exception)[:300]})
    return genre


# Cascades de Haar essayees pour le detecteur de repli, de la plus fiable a la
# plus permissive. Leur presence n'est jamais deduite de la version d'OpenCV :
# a partir d'OpenCV 5 le paquet ne les livre plus, et un test d'existence
# repond mieux qu'une hypothese.
CASCADES_HAAR = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
)


def sortir(dossier, statut, marqueur, code, detail, extra=None):
    """Aucun chemin de sortie sans marqueur, et un fichier temoin qui fait foi.

    Le greffon decide d'apres ce fichier et d'apres le code de retour, jamais
    d'apres la presence d'une chaine dans la sortie du processus : un texte
    cherche dans une sortie se retrouve aussi dans la trace d'erreur qui
    affiche la ligne de code qui le contient.
    """
    donnees = {"statut": statut, "marqueur": marqueur, "code": code,
               "detail": str(detail)[:2000]}
    if extra:
        donnees.update(extra)
    if dossier:
        try:
            chemin = os.path.join(dossier, "resultat.json")
            with open(chemin, "w", encoding="utf-8") as f:
                json.dump(donnees, f, indent=2)
        except Exception:
            pass
    print(marqueur + " " + str(detail)[:1500])
    sys.stdout.flush()
    sys.exit(code)


def charger_config():
    if len(sys.argv) < 2:
        print("[ERR_PARAMS] aucun fichier de parametres passe au worker")
        sys.exit(CODE_PARAMS)
    try:
        with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception as e:
        print("[ERR_PARAMS] parametres illisibles: " + str(e))
        sys.exit(CODE_PARAMS)


def poser_plafond_memoire(cfg, notes):
    """RLIMIT_AS, et seulement quand aucun materiel n'est demande.

    RLIMIT_AS limite l'espace d'adressage virtuel, pas la memoire residente :
    l'initialisation d'un contexte CUDA reserve couramment des dizaines de
    gigaoctets d'espace virtuel sans les toucher, et la limite tuerait le
    processus avant le premier calcul. Aucun equivalent n'est pose sous
    Windows : une protection dont la correction n'a pas ete prouvee par un
    test vaut moins que son absence.
    """
    if not cfg.get("plafond_memoire_actif", False):
        notes.append("plafond memoire non pose (materiel demande)")
        return 0
    if os.name == "nt":
        notes.append("plafond memoire non pose (Windows, aucune protection prouvee)")
        return 0
    try:
        import resource
    except Exception as e:
        notes.append("plafond memoire indisponible: " + str(e))
        return 0

    impose = int(cfg.get("plafond_memoire_octets", 0) or 0)
    if impose > 0:
        limite = impose
    else:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            notes.append("memoire physique inconnue, plafond non pose")
            return 0
        limite = int(total * float(cfg.get("plafond_memoire_ratio", 0.75)))
    if limite <= 0:
        return 0
    try:
        souple, dur = resource.getrlimit(resource.RLIMIT_AS)
        if dur not in (resource.RLIM_INFINITY,) and limite > dur:
            limite = dur
        resource.setrlimit(resource.RLIMIT_AS, (limite, dur))
        notes.append("plafond memoire pose a %d octets" % limite)
        return limite
    except Exception as e:
        notes.append("plafond memoire refuse par le systeme: " + str(e))
        return 0


def verifier_pivot(tableau, np, ou):
    """Format pivot impose entre deux etapes : tableau numpy, RGB, uint8.

    Entre chaque etape d'un pipeline, l'image revient a un format unique et
    documente. Chaque etape convertit ce pivot vers ce dont son modele a
    besoin, puis y revient avant de rendre la main. Un oubli ne genere aucune
    exception, juste un visage bleu : cette verification est la seule chose qui
    transforme ce silence en erreur.
    """
    if tableau is None:
        raise ValueError("pivot absent en sortie de " + ou)
    if not isinstance(tableau, np.ndarray):
        raise ValueError("pivot non numpy en sortie de " + ou)
    if tableau.dtype != np.uint8:
        raise ValueError("pivot de type %s au lieu de uint8 en sortie de %s"
                         % (tableau.dtype, ou))
    if tableau.ndim != 3 or tableau.shape[2] != 3:
        raise ValueError("pivot de forme %s au lieu de (h, w, 3) en sortie de %s"
                         % (tableau.shape, ou))
    return tableau


def normaliser_image(brut, np):
    """Retourne (pivot_rgb_uint8, alpha_source, est_gris, profondeur).

    GIMP exporte un PNG 16 bits des que l'image est en 16 ou 32 bits. Une
    lecture en IMREAD_UNCHANGED renvoie alors du uint16 : diviser par 255
    produirait des valeurs jusqu'a 257 et une image blanche, sans qu'aucune
    exception ne soit levee.
    """
    import cv2

    if brut.ndim == 2:
        brut = brut[:, :, np.newaxis]
    canaux = brut.shape[2]
    est_gris = canaux in (1, 2)
    a_alpha = canaux in (2, 4)

    profondeur = brut.dtype
    if profondeur == np.uint16:
        pour_ia = (brut.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif profondeur == np.uint8:
        pour_ia = brut
    else:
        maxi = float(brut.max()) if brut.size else 1.0
        echelle = 255.0 / maxi if maxi > 0 else 1.0
        pour_ia = (brut.astype(np.float32) * echelle).clip(0, 255).astype(np.uint8)

    if est_gris:
        gris = pour_ia[:, :, 0]
        pivot = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)
    else:
        # cv2.imread rend du BGR : c'est ici, et nulle part ailleurs, que la
        # conversion vers le pivot RGB a lieu.
        pivot = cv2.cvtColor(pour_ia[:, :, :3], cv2.COLOR_BGR2RGB)

    alpha = brut[:, :, canaux - 1] if a_alpha else None
    return np.ascontiguousarray(pivot), alpha, est_gris, profondeur


FOURNISSEURS_DISPONIBLES = []


def fournisseurs_effectifs(demandes, ort):
    """Croiser la demande avec les fournisseurs reellement disponibles : le
    moteur refuse une demande qu'il ne peut pas satisfaire.

    Ce que le moteur declare disponible est note pour le rapport : le greffon
    doit pouvoir joindre l'etat des deux couches a son diagnostic, plutot que
    de conditionner celui-ci a la detection dont il constate l'echec.
    """
    try:
        disponibles = list(ort.get_available_providers())
    except Exception:
        disponibles = ["CPUExecutionProvider"]
    for nom in disponibles:
        if nom not in FOURNISSEURS_DISPONIBLES:
            FOURNISSEURS_DISPONIBLES.append(nom)
    retenus = [p for p in demandes if p in disponibles]
    if "CPUExecutionProvider" not in retenus:
        retenus.append("CPUExecutionProvider")
    return retenus, disponibles


def materiel_lisible(fournisseur):
    table = {
        "CUDAExecutionProvider": "GPU NVIDIA",
        "ROCMExecutionProvider": "GPU AMD",
        "DmlExecutionProvider": "GPU DirectML",
        "CoreMLExecutionProvider": "GPU Apple",
        "CPUExecutionProvider": "processeur",
    }
    return table.get(fournisseur, fournisseur)


def ouvrir_session(chemin, fournisseurs, ort):
    """Session configuree pour rendre reellement la memoire.

    Par defaut onnxruntime conserve ses allocations pour accelerer les
    inferences successives. C'est parfait pour un flux video, et mortel pour un
    greffon qui doit passer la main a un autre modele juste apres.
    """
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    try:
        return ort.InferenceSession(chemin, sess_options=options,
                                    providers=fournisseurs)
    except Exception as e:
        raise ErreurModele("chargement de %s impossible: %s"
                           % (os.path.basename(chemin), e))


def purger_memoire(pause_s):
    """Collecte, vidage du cache CUDA, puis pause avant le modele suivant.

    Cette fonction ne prend volontairement pas la session en argument. Passer
    un objet a une fonction qui fait "del" dessus ne supprime que la reference
    locale de la fonction : celle de l'appelant survit, et l'objet reste en
    vie pendant toute la purge et toute la pause. L'appelant doit donc poser
    lui-meme sa reference a None, puis appeler ceci - c'est le seul ordre qui
    rende reellement la memoire avant de charger le modele suivant.

    La pause laisse au pilote le temps de desallouer : CUDA utilise un
    allocateur paresseux, et sans elle le modele suivant peut echouer a
    s'instancier sur un faux manque de memoire.
    """
    gc.collect()
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        if pause_s > 0:
            time.sleep(pause_s)
    except Exception:
        pass


def normaliser_tenseur(rgb_u8, normalisation, np):
    """RGB uint8 -> NCHW float32 normalise. x / 255, puis (x - moyenne) / ecart."""
    moyenne, ecart = float(normalisation[0]), float(normalisation[1])
    x = rgb_u8.astype(np.float32) / 255.0
    x = (x - moyenne) / (ecart if ecart else 1.0)
    x = np.transpose(x, (2, 0, 1))
    return np.ascontiguousarray(x[np.newaxis, ...], dtype=np.float32)


def denormaliser_tenseur(sortie, normalisation, np):
    """NCHW float32 normalise -> RGB uint8, en revenant au pivot."""
    moyenne, ecart = float(normalisation[0]), float(normalisation[1])
    tableau = np.asarray(sortie)
    while tableau.ndim > 3:
        tableau = tableau[0]
    if tableau.ndim != 3:
        raise ValueError("sortie de modele de forme inattendue: %s"
                         % (np.asarray(sortie).shape,))
    if tableau.shape[0] in (1, 3) and tableau.shape[0] < tableau.shape[2]:
        tableau = np.transpose(tableau, (1, 2, 0))
    if tableau.shape[2] == 1:
        tableau = np.repeat(tableau, 3, axis=2)
    tableau = tableau.astype(np.float32) * (ecart if ecart else 1.0) + moyenne
    return (tableau * 255.0).clip(0, 255).astype(np.uint8)


def derouler_sortie(sortie):
    """De nombreux exports renvoient un tuple ou un dictionnaire : derouler
    jusqu'au tenseur plutot que d'indexer a l'aveugle."""
    vu = 0
    while vu < 8:
        vu += 1
        if isinstance(sortie, dict):
            valeurs = list(sortie.values())
            if not valeurs:
                break
            sortie = valeurs[0]
            continue
        if isinstance(sortie, (list, tuple)):
            if not sortie:
                break
            sortie = sortie[0]
            continue
        break
    return sortie


def lettre_boite(pivot, taille, cv2, np):
    """Redimensionnement a rapport conserve, centre sur un fond neutre.

    Retourne (toile, echelle, dx, dy). Ces trois valeurs sont le contrat de
    conversion des coordonnees : les oublier place le masque ailleurs que le
    sujet, sans lever la moindre erreur.
    """
    hauteur, largeur = pivot.shape[:2]
    echelle = min(float(taille) / float(largeur), float(taille) / float(hauteur))
    nl = max(1, int(round(largeur * echelle)))
    nh = max(1, int(round(hauteur * echelle)))
    redim = cv2.resize(pivot, (nl, nh), interpolation=cv2.INTER_LINEAR)
    toile = np.full((taille, taille, 3), 114, dtype=np.uint8)
    dx = (taille - nl) // 2
    dy = (taille - nh) // 2
    toile[dy:dy + nh, dx:dx + nl] = redim
    return toile, echelle, dx, dy


def recouvrement(boite, autres, np):
    x1 = np.maximum(boite[0], autres[:, 0])
    y1 = np.maximum(boite[1], autres[:, 1])
    x2 = np.minimum(boite[0] + boite[2], autres[:, 0] + autres[:, 2])
    y2 = np.minimum(boite[1] + boite[3], autres[:, 1] + autres[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    aire = boite[2] * boite[3]
    aires = autres[:, 2] * autres[:, 3]
    union = aire + aires - inter
    return np.where(union > 0, inter / union, 0.0)


def supprimer_doublons(boites, scores, seuil, np):
    if len(boites) == 0:
        return []
    boites = np.asarray(boites, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    ordre = np.argsort(-scores)
    gardes = []
    while ordre.size > 0:
        indice = int(ordre[0])
        gardes.append(indice)
        if ordre.size == 1:
            break
        reste = ordre[1:]
        ious = recouvrement(boites[indice], boites[reste], np)
        ordre = reste[ious <= seuil]
    return gardes


def decoder_detections(brut, echelle, dx, dy, largeur, hauteur, cfg, np):
    """Transforme la sortie brute du detecteur en boites de l'espace image.

    La disposition de la sortie n'est pas la meme d'un export a l'autre :
    (1, C, N) pour les uns, (1, N, C) pour les autres. Elle se constate sur les
    dimensions, elle ne se suppose pas.
    """
    tableau = np.asarray(brut, dtype=np.float32)
    while tableau.ndim > 2:
        tableau = tableau[0]
    if tableau.ndim != 2:
        raise ValueError("sortie de detection de forme inattendue: %s"
                         % (np.asarray(brut).shape,))
    nb_classes = max(1, int(cfg["detection_nombre_classes"]))
    minimum = 4 + nb_classes
    # L'orientation se decide sur le nombre de colonnes exigible, pas sur un
    # simple "le plus petit axe est celui des canaux" : ce raccourci se trompe
    # des que le modele rend moins de boites que de colonnes, ce qui n'arrive
    # jamais en production mais arrive dans tout banc de test - et un banc qui
    # ne peut pas reproduire le cas ne prouve rien.
    if tableau.shape[1] < minimum <= tableau.shape[0]:
        tableau = tableau.T
    elif tableau.shape[0] < tableau.shape[1] and tableau.shape[0] >= minimum:
        tableau = tableau.T
    if tableau.shape[1] < minimum:
        raise ValueError("sortie de detection a %d colonnes, %d attendues au "
                         "minimum" % (tableau.shape[1], minimum))

    scores = tableau[:, 4:4 + nb_classes].max(axis=1)
    retenus = scores >= float(cfg["score_min_visage"])
    tableau = tableau[retenus]
    scores = scores[retenus]
    if tableau.shape[0] == 0:
        return [], []

    cx, cy = tableau[:, 0], tableau[:, 1]
    bl, bh = tableau[:, 2], tableau[:, 3]
    # Retour de l'espace lettre-boite vers l'espace image : d'abord le
    # decalage de remplissage, ensuite seulement l'echelle.
    x1 = (cx - bl / 2.0 - dx) / echelle
    y1 = (cy - bh / 2.0 - dy) / echelle
    larg = bl / echelle
    haut = bh / echelle

    boites = []
    gardes = []
    for index in range(tableau.shape[0]):
        bx = max(0.0, float(x1[index]))
        by = max(0.0, float(y1[index]))
        bw = min(float(larg[index]), largeur - bx)
        bh2 = min(float(haut[index]), hauteur - by)
        if bw <= 1.0 or bh2 <= 1.0:
            continue
        boites.append([bx, by, bw, bh2])
        gardes.append(float(scores[index]))
    return boites, gardes


def detecteur_haar(cv2):
    """Cascade de Haar livree avec OpenCV, si elle l'est.

    Sa presence ne se deduit pas de la version installee : a partir d'OpenCV 5
    les fichiers XML ne sont plus distribues avec le paquet. On regarde.
    """
    try:
        import cv2.data
        base = cv2.data.haarcascades
    except Exception as e:
        return None, "cv2.data indisponible (%s)" % e
    for nom in CASCADES_HAAR:
        chemin = os.path.join(base, nom)
        if not os.path.isfile(chemin):
            continue
        try:
            cascade = cv2.CascadeClassifier(chemin)
            if not cascade.empty():
                return cascade, nom
        except Exception:
            continue
    return None, "aucune cascade livree avec cette version d'OpenCV"


def filtrer_visages(boites, scores, cfg, largeur, hauteur, roi, np, notes):
    """Ecarte les visages trop petits, limite le nombre, restreint a la
    selection quand il y en a une.

    Le seuil de taille porte sur le cote, pas sur l'aire : un seuil d'aire est
    une hypothese sur la forme du sujet en plus d'une hypothese sur sa taille.
    """
    cote_min = float(cfg["cote_min_visage_ratio"]) * min(largeur, hauteur)
    retenus = []
    for boite, score in zip(boites, scores):
        if min(boite[2], boite[3]) < cote_min:
            continue
        if roi:
            cx = boite[0] + boite[2] / 2.0
            cy = boite[1] + boite[3] / 2.0
            rx, ry, rl, rh = roi
            if not (rx <= cx <= rx + rl and ry <= cy <= ry + rh):
                continue
        retenus.append((boite, score))
    ecartes = len(boites) - len(retenus)
    if ecartes > 0:
        notes.append("%d visage(s) ecarte(s) (trop petits ou hors selection)"
                     % ecartes)
    retenus.sort(key=lambda item: -item[1])
    maximum = max(1, int(cfg["visages_max"]))
    if len(retenus) > maximum:
        notes.append("%d visage(s) au-dela du maximum de %d ignore(s)"
                     % (len(retenus) - maximum, maximum))
        retenus = retenus[:maximum]
    return [item[0] for item in retenus], [item[1] for item in retenus]


def detecter_visages_yunet(pivot, chemin, cfg, np, cv2, notes):
    """Detection par YuNet, pilotee par cv2.FaceDetectorYN.

    OpenCV fait lui-meme le pretraitement et le decodage : il n'y a donc ni
    mise en lettre-boite ni lecture de sortie brute a ecrire ici, c'est-a-dire
    ni l'un ni l'autre a se tromper. Restent deux contrats a respecter, et ce
    sont les deux seuls endroits ou ce chemin peut echouer en silence :

    - l'entree se donne en BGR, alors que le pivot est en RGB. Une inversion
      ne leve rien : elle fait baisser le score et deplace legerement la boite.
      Le banc de test ne sait pas la distinguer, et le dit.
    - l'image est reduite avant detection, donc les boites rendues sont dans
      l'espace reduit. Les remettre a l'echelle est indispensable, et cette
      conversion-la est couverte par un test geometrique sur le vrai modele.
    """
    hauteur, largeur = pivot.shape[:2]
    cote_max = max(64, int(cfg["yunet_cote_max"]))
    echelle = min(1.0, float(cote_max) / float(max(hauteur, largeur)))
    lt = max(1, int(round(largeur * echelle)))
    ht = max(1, int(round(hauteur * echelle)))
    if echelle < 1.0:
        reduit = cv2.resize(pivot, (lt, ht), interpolation=cv2.INTER_AREA)
    else:
        reduit = pivot
    bgr = cv2.cvtColor(reduit, cv2.COLOR_RGB2BGR)

    fabrique = getattr(cv2, "FaceDetectorYN", None)
    creer = getattr(fabrique, "create", None) if fabrique else None
    if creer is None:
        creer = getattr(cv2, "FaceDetectorYN_create", None)
    if creer is None:
        raise ErreurModele("cette version d'OpenCV n'expose pas FaceDetectorYN")

    detecteur = creer(chemin, "", (lt, ht), float(cfg["score_min_visage"]),
                      float(cfg["nms_recouvrement_max"]),
                      max(1, int(cfg["visages_max"])))
    try:
        detecteur.setInputSize((lt, ht))
        try:
            _, trouves = detecteur.detect(bgr)
        except MemoryError:
            raise
        except Exception as e:
            raise ErreurInference("calcul refuse par le moteur: " + str(e))
    finally:
        detecteur = None
        purger_memoire(float(cfg["pause_entre_modeles_s"]))

    boites = []
    scores = []
    if trouves is None:
        return boites, scores
    for ligne in np.asarray(trouves, dtype=np.float32):
        if ligne.shape[0] < 4:
            continue
        # Retour vers l'espace de l'image d'origine. L'oublier placerait le
        # traitement sur le quart superieur gauche de toute grande photo.
        bx = max(0.0, float(ligne[0]) / echelle)
        by = max(0.0, float(ligne[1]) / echelle)
        bl = min(float(ligne[2]) / echelle, largeur - bx)
        bh = min(float(ligne[3]) / echelle, hauteur - by)
        if bl <= 1.0 or bh <= 1.0:
            continue
        boites.append([bx, by, bl, bh])
        scores.append(float(ligne[14]) if ligne.shape[0] > 14 else 0.0)
    notes.append("YuNet a examine l'image en %dx%d (facteur %.3f)"
                 % (lt, ht, echelle))
    return boites, scores


def detecter_visages(pivot, cfg, modele, fournisseurs, np, cv2, notes):
    """Retourne (boites, scores, moteur, fournisseur).

    Trois chemins, du meilleur au plus fruste : le modele ONNX, la cascade de
    Haar livree avec OpenCV, puis la selection de l'utilisateur prise pour un
    unique visage. Le dernier ne trouve rien, mais il permet a l'anonymisation
    d'aboutir sur un poste sans reseau, ce qui vaut mieux qu'un message
    d'erreur a la place d'un calque.
    """
    hauteur, largeur = pivot.shape[:2]
    roi = cfg.get("roi_visages") or None

    # Un modele YOLO present sur le disque l'a ete deliberement : le greffon ne
    # va jamais le chercher de lui-meme. Il passe donc devant YuNet, qu'il
    # telecharge. Un choix explicite ne se satisfait pas d'un substitut
    # silencieux.
    if modele:
        try:
            import onnxruntime as ort
            demandes, disponibles = fournisseurs_effectifs(fournisseurs, ort)
            session = ouvrir_session(modele, demandes, ort)
            fournisseur = (session.get_providers() or demandes)[0]
            taille = int(cfg["taille_entree_detection"])
            toile, echelle, dx, dy = lettre_boite(pivot, taille, cv2, np)
            tenseur = normaliser_tenseur(toile, cfg["detection_normalisation"], np)
            nom_entree = session.get_inputs()[0].name
            try:
                brut = derouler_sortie(session.run(None, {nom_entree: tenseur}))
            except MemoryError:
                raise
            except Exception as e:
                raise ErreurInference("calcul refuse par le moteur: " + str(e))
            finally:
                session = None
                purger_memoire(float(cfg["pause_entre_modeles_s"]))
            boites, scores = decoder_detections(brut, echelle, dx, dy, largeur,
                                                hauteur, cfg, np)
            indices = supprimer_doublons(boites, scores,
                                         float(cfg["nms_recouvrement_max"]), np)
            boites = [boites[i] for i in indices]
            scores = [scores[i] for i in indices]
            boites, scores = filtrer_visages(boites, scores, cfg, largeur,
                                             hauteur, roi, np, notes)
            if boites:
                return boites, scores, os.path.basename(modele), fournisseur
            notes.append("le modele de detection n'a trouve aucun visage, "
                         "essai du detecteur suivant")
        except MemoryError:
            raise
        except Exception as e:
            genre = noter_echec("detection", e)
            notes.append("detection ONNX en echec (%s: %s), passage au "
                         "detecteur suivant" % (genre, str(e)[:200]))

    chemin_yunet = cfg["modeles"].get("detection_yunet")
    if chemin_yunet:
        try:
            boites, scores = detecter_visages_yunet(pivot, chemin_yunet, cfg,
                                                    np, cv2, notes)
            boites, scores = filtrer_visages(boites, scores, cfg, largeur,
                                             hauteur, roi, np, notes)
            if boites:
                return (boites, scores, os.path.basename(chemin_yunet),
                        "processeur")
            notes.append("YuNet n'a trouve aucun visage, essai de la cascade "
                         "de Haar")
        except MemoryError:
            raise
        except Exception as e:
            genre = noter_echec("detection_yunet", e)
            notes.append("YuNet en echec (%s: %s), repli sur la cascade de "
                         "Haar" % (genre, str(e)[:200]))

    cascade, detail = detecteur_haar(cv2)
    if cascade is not None:
        gris = cv2.cvtColor(pivot, cv2.COLOR_RGB2GRAY)
        cote_min = int(float(cfg["cote_min_visage_ratio"]) * min(largeur, hauteur))
        trouves = cascade.detectMultiScale(
            gris, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(8, cote_min), max(8, cote_min)))
        boites = [[float(x), float(y), float(w), float(h)] for x, y, w, h in trouves]
        # La cascade ne rend pas de score. Le nom du calque doit dire d'ou
        # vient le resultat : ce 0.0 n'est pas une confiance, c'est une
        # absence de confiance, et il est affiche comme tel.
        scores = [0.0] * len(boites)
        boites, scores = filtrer_visages(boites, scores, cfg, largeur, hauteur,
                                         roi, np, notes)
        if boites:
            return boites, scores, "cascade " + detail, "processeur"
        notes.append("la cascade de Haar n'a trouve aucun visage")
    else:
        notes.append("cascade de Haar indisponible : " + detail)

    if roi:
        rx, ry, rl, rh = roi
        notes.append("aucun detecteur n'a abouti : la selection est traitee "
                     "comme un visage unique")
        return ([[float(rx), float(ry), float(rl), float(rh)]], [0.0],
                "selection de l'utilisateur", "processeur")
    return [], [], "aucun", "processeur"


def boite_avec_marge(boite, cfg, largeur, hauteur):
    """Recadrage sur la zone d'interet avec marge proportionnelle.

    Les modeles de restauration attendent le menton et le front, que la boite
    de detection coupe systematiquement.
    """
    x, y, bl, bh = boite
    marge = max(float(cfg["marge_visage_ratio"]) * max(bl, bh),
                float(cfg["marge_visage_min_px"]))
    x1 = int(max(0, round(x - marge)))
    y1 = int(max(0, round(y - marge)))
    x2 = int(min(largeur, round(x + bl + marge)))
    y2 = int(min(hauteur, round(y + bh + marge)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def masque_de_fondu(hauteur, largeur, plume_ratio, np):
    """Masque de fusion nul sur l'anneau exterieur.

    Hors de la zone fondue, l'ecart pixel avec l'original doit etre exactement
    nul : c'est ce qui rend verifiable la promesse que le greffon ne touche
    qu'aux visages.
    """
    plume = max(1.0, round(min(hauteur, largeur) * float(plume_ratio)))
    colonnes = np.minimum(np.arange(largeur), largeur - 1 - np.arange(largeur))
    lignes = np.minimum(np.arange(hauteur), hauteur - 1 - np.arange(hauteur))
    distance = np.minimum(colonnes[np.newaxis, :].astype(np.float32),
                          lignes[:, np.newaxis].astype(np.float32))
    return np.clip(distance / plume, 0.0, 1.0).astype(np.float32)


def anonymiser_vignette(vignette, cfg, cv2, np):
    """Flou gaussien ou mosaique, sans aucune IA.

    Les rayons sont exprimes en fraction du cote de la vignette : un rayon fixe
    laisse un visage parfaitement lisible sur une photo haute definition.
    """
    hauteur, largeur = vignette.shape[:2]
    cote = min(hauteur, largeur)
    if int(cfg["anonymisation_mode"]) == 1:
        bloc = max(2, int(round(cote * float(cfg["anonyme_pixels_ratio"]))))
        petite = cv2.resize(vignette,
                            (max(1, largeur // bloc), max(1, hauteur // bloc)),
                            interpolation=cv2.INTER_LINEAR)
        return cv2.resize(petite, (largeur, hauteur),
                          interpolation=cv2.INTER_NEAREST)
    noyau = int(round(cote * float(cfg["anonyme_flou_ratio"])))
    noyau = max(3, noyau + (1 - noyau % 2))
    return cv2.GaussianBlur(vignette, (noyau, noyau), 0)


def rehausser_vignette(vignette, cfg, cv2, np):
    """Repli sans IA de l'amelioration : masque flou puis lissage.

    Ce n'est pas une restauration - aucun detail n'est invente - et le nom du
    calque le dit en toutes lettres.
    """
    force = float(cfg["rehaussement_force"])
    rayon = max(1, int(cfg["rehaussement_rayon"]))
    flou = cv2.GaussianBlur(vignette, (0, 0), rayon)
    net = cv2.addWeighted(vignette, 1.0 + force, flou, -force, 0)
    try:
        net = cv2.bilateralFilter(net, 5, 40, 40)
    except Exception:
        pass
    return net.astype(np.uint8)


def appliquer_modele_vignettes(vignettes, chemin, taille, normalisation, cfg,
                               fournisseurs, np, cv2, vecteur=None):
    """Applique un modele a toutes les vignettes, avec une seule session.

    Une session par vignette rechargerait le modele autant de fois qu'il y a
    de visages. Une session pour tout le lot, puis dechargement : l'empreinte
    a l'instant T reste celle d'un seul modele.
    """
    import onnxruntime as ort

    demandes, disponibles = fournisseurs_effectifs(fournisseurs, ort)
    session = ouvrir_session(chemin, demandes, ort)
    fournisseur = (session.get_providers() or demandes)[0]
    entrees = session.get_inputs()
    nom_image = entrees[0].name
    nom_vecteur = entrees[1].name if len(entrees) > 1 else None
    resultats = []
    try:
        for vignette in vignettes:
            hauteur, largeur = vignette.shape[:2]
            petite = cv2.resize(vignette, (taille, taille),
                                interpolation=cv2.INTER_LINEAR)
            alimentation = {nom_image: normaliser_tenseur(petite, normalisation, np)}
            if nom_vecteur is not None and vecteur is not None:
                alimentation[nom_vecteur] = vecteur
            try:
                sortie = derouler_sortie(session.run(None, alimentation))
            except MemoryError:
                raise
            except Exception as e:
                raise ErreurInference("calcul refuse par le moteur: " + str(e))
            rgb = denormaliser_tenseur(sortie, normalisation, np)
            resultats.append(cv2.resize(rgb, (largeur, hauteur),
                                        interpolation=cv2.INTER_LINEAR))
            sortie = None
            rgb = None
    finally:
        session = None
        purger_memoire(float(cfg["pause_entre_modeles_s"]))
    return resultats, os.path.basename(chemin), fournisseur


def etape_sourire(vignettes, chemin, cfg, fournisseurs, np, cv2):
    """Modification d'expression sur les vignettes.

    Le vecteur d'attributs n'est pas devine par le worker : sa taille et
    l'indice de l'attribut viennent du fichier de parametres, parce qu'ils
    dependent de l'export et de rien d'autre.
    """
    taille_vecteur = max(1, int(cfg["sourire_vecteur_taille"]))
    indice = min(max(0, int(cfg["sourire_indice_attribut"])), taille_vecteur - 1)
    vecteur = np.zeros((1, taille_vecteur), dtype=np.float32)
    vecteur[0, indice] = float(cfg["sourire_intensite"])
    return appliquer_modele_vignettes(
        vignettes, chemin, int(cfg["sourire_taille_entree"]),
        cfg["sourire_normalisation"], cfg, fournisseurs, np, cv2, vecteur)


def etape_amelioration(vignettes, chemin, cfg, fournisseurs, np, cv2):
    return appliquer_modele_vignettes(
        vignettes, chemin, int(cfg["amelioration_taille_entree"]),
        cfg["amelioration_normalisation"], cfg, fournisseurs, np, cv2)


def etape_colorisation(pivot, chemin, cfg, fournisseurs, np, cv2):
    """Colorisation de l'image entiere, chrominance seule.

    La luminance du resultat est reprise de l'original : le modele ne fournit
    que la couleur. Ce choix donne une propriete verifiable - la luminance ne
    bouge pas - la ou une sortie RGB reprise telle quelle peut blanchir ou
    assombrir l'image sans que rien ne le signale.
    """
    import onnxruntime as ort

    hauteur, largeur = pivot.shape[:2]
    taille = int(cfg["colorisation_taille_entree"])
    normalisation = cfg["colorisation_normalisation"]

    gris = cv2.cvtColor(pivot, cv2.COLOR_RGB2GRAY)
    entree_rgb = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)
    petite = cv2.resize(entree_rgb, (taille, taille), interpolation=cv2.INTER_LINEAR)

    demandes, disponibles = fournisseurs_effectifs(fournisseurs, ort)
    session = ouvrir_session(chemin, demandes, ort)
    fournisseur = (session.get_providers() or demandes)[0]
    try:
        nom_entree = session.get_inputs()[0].name
        try:
            sortie = derouler_sortie(session.run(
                None, {nom_entree: normaliser_tenseur(petite, normalisation, np)}))
        except MemoryError:
            raise
        except Exception as e:
            raise ErreurInference("calcul refuse par le moteur: " + str(e))
        colore_petit = denormaliser_tenseur(sortie, normalisation, np)
        sortie = None
    finally:
        session = None
        purger_memoire(float(cfg["pause_entre_modeles_s"]))

    colore = cv2.resize(colore_petit, (largeur, hauteur),
                        interpolation=cv2.INTER_LINEAR)
    lab_modele = cv2.cvtColor(colore, cv2.COLOR_RGB2LAB)
    lab_origine = cv2.cvtColor(entree_rgb, cv2.COLOR_RGB2LAB)
    lab_modele[:, :, 0] = lab_origine[:, :, 0]
    resultat = cv2.cvtColor(lab_modele, cv2.COLOR_LAB2RGB)
    return np.ascontiguousarray(resultat), os.path.basename(chemin), fournisseur


def arbitrer_operations(operations, notes):
    """Table de preseance appliquee une seconde fois, cote worker.

    Le greffon expurge deja les intentions contradictoires avant d'ecrire le
    fichier de parametres. Ce controle est redondant, et c'est voulu : il dit
    aussi, par le drapeau qu'il rend, si le greffon a fait son travail. Un test
    qui desactive l'arbitrage du greffon voit ce drapeau passer a vrai, ce qui
    le rend discriminant pour chacun des deux mecanismes pris separement.
    """
    operations = list(operations)
    if "anonymiser" not in operations:
        return operations, False
    retires = [op for op in ("sourire", "ameliorer") if op in operations]
    if not retires:
        return operations, False
    for op in retires:
        operations.remove(op)
    notes.append("preseance appliquee par le worker : %s ignoree(s) car "
                 "l'anonymisation est prioritaire" % ", ".join(retires))
    return operations, True


def melanger_canaux(origine, nouveau, masque, np):
    """Composition sur le seul masque, avec identite stricte hors fondu."""
    m = masque
    melange = (origine.astype(np.float32) * (1.0 - m)
               + nouveau.astype(np.float32) * m)
    return np.where(m > 0, np.rint(melange), origine.astype(np.float32))


def run():
    debut = time.time()
    cfg = charger_config()
    dossier = cfg.get("dossier_sortie") or ""
    notes = []
    degradations = []
    moteurs = []

    try:
        import numpy as np
    except Exception as e:
        sortir(dossier, "erreur_import", "[ERR_IMPORT]", CODE_IMPORT,
               "numpy indisponible: " + str(e))
        return
    try:
        import cv2
    except Exception as e:
        sortir(dossier, "erreur_import", "[ERR_IMPORT]", CODE_IMPORT,
               "opencv indisponible: " + str(e))
        return

    # Le plafond se pose apres les imports : l'espace d'adressage que numpy et
    # opencv reservent en se chargeant n'est pas de la memoire de travail, et
    # le compter dedans ferait echouer le worker avant son premier calcul.
    try:
        poser_plafond_memoire(cfg, notes)
    except Exception as e:
        notes.append("plafond memoire non pose: " + str(e))

    try:
        chemin_image = cfg["image"]
        brut = cv2.imread(chemin_image, cv2.IMREAD_UNCHANGED)
        if brut is None:
            sortir(dossier, "erreur_image", "[ERR_IMAGE]", CODE_IMAGE,
                   "image illisible: " + str(chemin_image))
            return
        if brut.ndim == 2:
            # Une image en niveaux de gris sans transparence est relue en deux
            # dimensions. La composition, elle, indexe toujours un troisieme
            # axe : c'est ici qu'il faut le poser, et pas seulement dans la
            # copie locale de normaliser_image.
            brut = brut[:, :, np.newaxis]

        pivot, alpha_source, est_gris, profondeur = normaliser_image(brut, np)
        verifier_pivot(pivot, np, "lecture de l'image")
        hauteur, largeur = pivot.shape[:2]
        base_avant_visages = pivot

        operations, preseance_worker = arbitrer_operations(
            cfg["operations"], notes)
        modeles = cfg["modeles"]
        fournisseurs = cfg["fournisseurs"]

        # --- 1. Colorisation, sur l'image entiere -----------------------------
        colorise = False
        if "coloriser" in operations:
            chemin = modeles.get("colorisation")
            if chemin:
                try:
                    pivot, moteur, fournisseur = etape_colorisation(
                        pivot, chemin, cfg, fournisseurs, np, cv2)
                    verifier_pivot(pivot, np, "colorisation")
                    base_avant_visages = pivot
                    colorise = True
                    moteurs.append(["colorisation", moteur,
                                    materiel_lisible(fournisseur)])
                except MemoryError:
                    raise
                except ValueError as e:
                    sortir(dossier, "erreur_pivot", "[ERR_PIVOT]", CODE_PIVOT,
                           "format pivot rompu: " + str(e))
                    return
                except Exception as e:
                    noter_echec("colorisation", e)
                    degradations.append("colorisation ignoree : " + str(e)[:200])
            else:
                degradations.append(
                    "colorisation ignoree : aucun modele utilisable")

        # --- 2. Detection des visages ----------------------------------------
        operations_faciales = [op for op in operations
                               if op in ("anonymiser", "sourire", "ameliorer")]
        boites, scores = [], []
        moteur_detection = "aucun"
        fournisseur_detection = "processeur"
        if operations_faciales:
            boites, scores, moteur_detection, fournisseur_detection = \
                detecter_visages(pivot, cfg, modeles.get("detection"),
                                 fournisseurs, np, cv2, notes)
            moteurs.append(["detection", moteur_detection,
                            materiel_lisible(fournisseur_detection)])

        rectangles = []
        vignettes = []
        scores_retenus = []
        for boite, score in zip(boites, scores):
            rect = boite_avec_marge(boite, cfg, largeur, hauteur)
            if rect is None:
                continue
            x, y, bl, bh = rect
            rectangles.append(rect)
            scores_retenus.append(score)
            vignettes.append(np.ascontiguousarray(pivot[y:y + bh, x:x + bl]))

        if not vignettes and not colorise:
            # Le marqueur doit designer la cause, pas le symptome : un modele
            # absent et une image sans visage ne se reparent pas de la meme
            # facon, et c'est ce message que l'utilisateur lira.
            genres = [echec["genre"] for echec in ECHECS_MODELES]
            detail_echecs = " | ".join(
                "%s (%s): %s" % (e["role"], e["genre"], e["detail"])
                for e in ECHECS_MODELES)
            extra = {"notes": notes, "degradations": degradations,
                     "echecs_modeles": ECHECS_MODELES}
            if "modele" in genres:
                sortir(dossier, "erreur_modele", "[ERR_MODELE]", CODE_MODELE,
                       "aucun modele utilisable pour les operations demandees : "
                       + detail_echecs, extra)
            elif "inference" in genres:
                sortir(dossier, "erreur_inference", "[ERR_INFERENCE]",
                       CODE_INFERENCE,
                       "l'inference a echoue sur toutes les operations "
                       "demandees : " + detail_echecs, extra)
            else:
                sortir(dossier, "aucun_visage", "[ERR_AUCUN_VISAGE]",
                       CODE_AUCUN_VISAGE,
                       "aucun visage detecte et aucune autre operation a "
                       "appliquer (moteur de detection : %s)" % moteur_detection,
                       extra)
            return

        # --- 3. Operations faciales, un modele a la fois ----------------------
        if vignettes:
            # Les trois etapes se declenchent independamment : l'exclusion
            # de l'anonymisation avec les deux autres est le travail de la
            # table de preseance, et de rien d'autre. Un "else" cache ici
            # ferait le meme effet, mais il donnerait un troisieme mecanisme
            # redondant qu'aucun test ne saurait distinguer des deux premiers -
            # et deux des correctifs de cette serie se sont revele intestables
            # pour exactement cette raison.
            if "anonymiser" in operations:
                vignettes = [anonymiser_vignette(v, cfg, cv2, np)
                             for v in vignettes]
                for index, vignette in enumerate(vignettes):
                    verifier_pivot(vignette, np, "anonymisation")
                moteurs.append(["anonymisation", "OpenCV", "processeur"])
            if "sourire" in operations:
                chemin = modeles.get("sourire")
                if chemin:
                    try:
                        vignettes, moteur, fournisseur = etape_sourire(
                            vignettes, chemin, cfg, fournisseurs, np, cv2)
                        for vignette in vignettes:
                            verifier_pivot(vignette, np, "sourire")
                        moteurs.append(["sourire", moteur,
                                        materiel_lisible(fournisseur)])
                    except MemoryError:
                        raise
                    except ValueError as e:
                        sortir(dossier, "erreur_pivot", "[ERR_PIVOT]",
                               CODE_PIVOT, "format pivot rompu: " + str(e))
                        return
                    except Exception as e:
                        noter_echec("sourire", e)
                        degradations.append("sourire ignore : " + str(e)[:200])
                else:
                    degradations.append(
                        "sourire ignore : aucun modele utilisable")

            if "ameliorer" in operations:
                chemin = modeles.get("amelioration")
                fait = False
                if chemin:
                    try:
                        vignettes, moteur, fournisseur = etape_amelioration(
                            vignettes, chemin, cfg, fournisseurs, np, cv2)
                        for vignette in vignettes:
                            verifier_pivot(vignette, np, "amelioration")
                        moteurs.append(["amelioration", moteur,
                                        materiel_lisible(fournisseur)])
                        fait = True
                    except MemoryError:
                        raise
                    except ValueError as e:
                        sortir(dossier, "erreur_pivot", "[ERR_PIVOT]",
                               CODE_PIVOT, "format pivot rompu: " + str(e))
                        return
                    except Exception as e:
                        noter_echec("amelioration", e)
                        degradations.append(
                            "amelioration par IA en echec (%s), "
                            "rehaussement sans IA applique" % str(e)[:160])
                if not fait:
                    vignettes = [rehausser_vignette(v, cfg, cv2, np)
                                 for v in vignettes]
                    moteurs.append(["amelioration",
                                    "sans IA (rehaussement OpenCV)",
                                    "processeur"])
                    if chemin is None:
                        degradations.append(
                            "amelioration sans IA : aucun modele utilisable")

        # --- 4. Composition, a la profondeur de la source ---------------------
        # Les vignettes ne sont jamais fusionnees dans le pivot en cours de
        # route : elles le sont une seule fois, ici, pour que la composition
        # n'applique qu'un seul fondu et que l'identite hors zone traitee reste
        # demontrable.
        final_rgb = base_avant_visages.copy()
        masque_total = np.zeros((hauteur, largeur), dtype=np.float32)
        for rect, vignette in zip(rectangles, vignettes):
            x, y, bl, bh = rect
            masque = masque_de_fondu(bh, bl, cfg["fusion_plume_ratio"], np)
            final_rgb[y:y + bh, x:x + bl] = vignette
            zone = masque_total[y:y + bh, x:x + bl]
            masque_total[y:y + bh, x:x + bl] = np.maximum(zone, masque)

        seize_bits = profondeur == np.uint16
        facteur = 257.0 if seize_bits else 1.0
        plafond = 65535 if seize_bits else 255
        profondeur_sortie = np.uint16 if seize_bits else np.uint8
        profondeur_connue = seize_bits or profondeur == np.uint8
        gris_sortie = est_gris and not colorise

        try:
            if gris_sortie:
                nouveau = (cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY)
                           .astype(np.float32) * facteur)
                if profondeur_connue:
                    origine = brut[:, :, 0].astype(np.float32)
                else:
                    origine = (cv2.cvtColor(base_avant_visages, cv2.COLOR_RGB2GRAY)
                               .astype(np.float32))
                canaux = melanger_canaux(origine, nouveau, masque_total, np)
                canaux = canaux[:, :, np.newaxis]
            else:
                nouveau = final_rgb[:, :, ::-1].astype(np.float32) * facteur
                if colorise or not profondeur_connue:
                    origine = (base_avant_visages[:, :, ::-1].astype(np.float32)
                               * facteur)
                else:
                    origine = brut[:, :, :3].astype(np.float32)
                canaux = melanger_canaux(origine, nouveau,
                                         masque_total[:, :, np.newaxis], np)
            canaux = np.clip(canaux, 0, plafond).astype(profondeur_sortie)

            if alpha_source is not None:
                alpha = alpha_source
                if not seize_bits and alpha.dtype == np.uint16:
                    alpha = (alpha.astype(np.float32) / 257.0).clip(0, 255)
                elif seize_bits and alpha.dtype == np.uint8:
                    alpha = alpha.astype(np.float32) * 257.0
                alpha = np.asarray(alpha).astype(profondeur_sortie)
                if canaux.shape[2] == 1:
                    # OpenCV n'ecrit pas de PNG gris + alpha. Pour ne pas
                    # perdre la transparence, la sortie passe en couleur, les
                    # trois canaux portant la meme valeur.
                    canaux = np.repeat(canaux, 3, axis=2)
                    notes.append("source grise avec transparence rendue en "
                                 "couleur (trois canaux identiques)")
                sortie_image = np.dstack([canaux, alpha])
            elif canaux.shape[2] == 1:
                sortie_image = canaux[:, :, 0]
            else:
                sortie_image = canaux

            chemin_sortie = os.path.join(dossier, "resultat.png")
            if not cv2.imwrite(chemin_sortie, sortie_image):
                raise IOError("cv2.imwrite a refuse d'ecrire " + chemin_sortie)
            if not os.path.isfile(chemin_sortie) or os.path.getsize(chemin_sortie) == 0:
                raise IOError("fichier de sortie vide: " + chemin_sortie)
        except MemoryError:
            raise
        except Exception as e:
            sortir(dossier, "erreur_ecriture", "[ERR_ECRITURE]", CODE_ECRITURE,
                   "ecriture du resultat impossible: " + str(e))
            return

        details_visages = []
        aire_image = float(max(1, largeur * hauteur))
        for index, rect in enumerate(rectangles):
            x, y, bl, bh = rect
            details_visages.append({
                "rect": [int(x), int(y), int(bl), int(bh)],
                "score": round(float(scores_retenus[index]), 4)
                if index < len(scores_retenus) else 0.0,
                "aire_ratio": round((float(bl) * float(bh)) / aire_image, 5)})

        for etape, moteur, materiel in moteurs:
            print("[INFO_MOTEUR] %s: %s" % (etape, moteur))
            print("[INFO_MATERIEL] %s: %s" % (etape, materiel))
        for ligne in degradations:
            print("[INFO_DEGRADATION] " + ligne)
        sys.stdout.flush()

        sortir(dossier, "succes", "[OK_RESULTAT]", CODE_OK,
               "%d visage(s) traite(s)" % len(rectangles),
               {"fichier": "resultat.png",
                "visages": details_visages,
                "nombre_visages": len(rectangles),
                "moteurs": moteurs,
                "moteur_detection": moteur_detection,
                "materiel_detection": materiel_lisible(fournisseur_detection),
                "operations": operations,
                "operations_demandees": cfg["operations"],
                "preseance_appliquee_par_worker": preseance_worker,
                "colorisation_appliquee": colorise,
                "degradations": degradations,
                "echecs_modeles": ECHECS_MODELES,
                "notes": notes,
                "profondeur_source": str(profondeur),
                "profondeur_sortie": str(np.dtype(profondeur_sortie)),
                "source_grise": bool(est_gris),
                "sortie_grise": bool(gris_sortie),
                "alpha_source": alpha_source is not None,
                "fournisseurs_demandes": fournisseurs,
                "fournisseurs_disponibles": list(FOURNISSEURS_DISPONIBLES),
                "pivot_espace": cfg["pivot_espace"],
                "pivot_type": cfg["pivot_type"],
                "duree_totale_s": round(time.time() - debut, 2)})

    except MemoryError as e:
        sortir(dossier, "erreur_memoire", "[ERR_MEMOIRE]", CODE_MEMOIRE,
               "memoire insuffisante: " + str(e), {"notes": notes})
    except ValueError as e:
        sortir(dossier, "erreur_pivot", "[ERR_PIVOT]", CODE_PIVOT,
               "format pivot ou contrat de modele rompu: " + str(e),
               {"notes": notes})
    except Exception as e:
        sortir(dossier, "erreur_inattendue", "[ERR_INATTENDU]", CODE_INATTENDU,
               str(e) + " | " + traceback.format_exc()[-800:], {"notes": notes})


if __name__ == "__main__":
    run()
'''


# ==============================================================================
# 10. EXECUTION DU WORKER
# ==============================================================================
CODES_WORKER = {
    0: "succes", 2: "erreur_params", 3: "erreur_import", 4: "erreur_image",
    5: "erreur_modele", 6: "erreur_pivot", 7: "erreur_inference",
    8: "aucun_visage", 9: "erreur_ecriture", 10: "erreur_memoire",
    11: "erreur_inattendue",
}


def executer_worker(python_venv, dossier_travail, texte_progression, env=None):
    """Lance le worker et retourne (statut, resultat, journal_texte).

    Le statut vient du fichier temoin resultat.json, corrobore par le code de
    retour. Aucune decision ne repose sur la presence d'une chaine dans la
    sortie du processus : la recherche de texte n'enrichit qu'un diagnostic
    deja etabli autrement.
    """
    script = os.path.join(dossier_travail, "worker.py")
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(WORKER_SCRIPT)
    try:
        os.chmod(script, 0o755)
    except Exception:
        pass

    chemin_log = os.path.join(dossier_travail, "worker.log")
    cfg = os.path.join(dossier_travail, "cfg.json")
    with open(chemin_log, "w", encoding="utf-8") as lf:
        proc = demarrer_processus([python_venv, script, cfg],
                                  env or clean_env(), lf)
        code, depasse, annule = attendre_processus(proc, DELAI_WORKER_S,
                                                   texte_progression)

    journal_texte = lire_journal(chemin_log)
    resultat = {}
    try:
        with open(os.path.join(dossier_travail, "resultat.json"), "r",
                  encoding="utf-8", errors="replace") as f:
            resultat = json.load(f)
    except Exception:
        resultat = {}

    if annule:
        return "annule", resultat, journal_texte
    if depasse:
        return "timeout", resultat, journal_texte
    statut = resultat.get("statut")
    if not statut:
        statut = CODES_WORKER.get(code, "erreur_inattendue" if code else "succes")
    return statut, resultat, journal_texte


# ==============================================================================
# 11. ARBITRAGE DES INTENTIONS
#     Un greffon qui propose des options cumulatives ne peut pas se contenter
#     de lancer les traitements les uns apres les autres : il doit regler les
#     conflits logiques avant de transmettre quoi que ce soit au worker.
# ==============================================================================
LIBELLES_OPERATIONS = {
    "coloriser": "colorisation",
    "anonymiser": "anonymisation",
    "sourire": "sourire",
    "ameliorer": "amelioration",
}


def arbitrer_intentions(demandees, journal_conflits):
    """Applique la table de preseance et retourne la liste expurgee.

    L'anonymisation annule et remplace toute autre modification faciale : il
    est absurde d'ameliorer ou de faire sourire un visage qu'on vient de
    flouter, et l'ordre dans lequel les deux s'appliqueraient tiendrait du
    hasard. La colorisation, qui porte sur l'image entiere et sur une autre
    dimension, n'entre en conflit avec rien.

    Le conflit est resolu ici, avant l'ecriture du fichier de parametres, et il
    laisse une trace explicite dans le journal comme dans le rapport final.
    """
    retenues = [op for op in demandees if op in LIBELLES_OPERATIONS]
    if "anonymiser" not in retenues:
        return retenues
    ecartees = [op for op in PRESEANCE_FACIALE
                if op != "anonymiser" and op in retenues]
    for op in ecartees:
        retenues.remove(op)
        message = ("%s ignoree car l'anonymisation est prioritaire"
                   % LIBELLES_OPERATIONS[op].capitalize())
        journal_conflits.append(message)
        journal("arbitrage: " + message)
    return retenues


def roles_necessaires(operations):
    """Modeles a resoudre pour les operations retenues.

    La detection n'est demandee que si une operation faciale l'exige : une
    colorisation seule ne doit declencher aucun telechargement de detecteur.
    """
    roles = []
    if any(op in ("anonymiser", "sourire", "ameliorer") for op in operations):
        # Les deux detecteurs sont resolus : YuNet est celui que le greffon
        # sait aller chercher, l'autre n'est utilise que s'il est deja la.
        roles.append(ROLE_DETECTION_YUNET)
        roles.append(ROLE_DETECTION)
    if "sourire" in operations:
        roles.append(ROLE_SOURIRE)
    if "ameliorer" in operations:
        roles.append(ROLE_AMELIORATION)
    if "coloriser" in operations:
        roles.append(ROLE_COLORISATION)
    return roles


def etiquette_resultat(resultat):
    """Nom du calque : le moteur reellement employe et le materiel de calcul,
    en termes comprehensibles.

    Le journal du worker disparait avec le dossier temporaire ; le nom du
    calque est donc la seule trace persistante. "IA, CPU" ne dit rien -
    "anonymisation OpenCV, detection yolov8n-face.onnx sur processeur" se lit.
    Ces valeurs sont constatees par le worker, jamais deduites par le greffon.
    """
    moteurs = resultat.get("moteurs") or []
    morceaux = []
    for entree in moteurs:
        try:
            etape, moteur, materiel = entree[0], entree[1], entree[2]
        except Exception:
            continue
        morceaux.append("%s %s sur %s" % (etape, moteur, materiel))
    if not morceaux:
        return "aucun traitement constate"
    return ", ".join(morceaux)


def materiel_constate(resultat):
    for entree in (resultat.get("moteurs") or []):
        try:
            if entree[2] and entree[2] != "processeur":
                return entree[2]
        except Exception:
            continue
    return "processeur"


def signaler_ecart_materiel(gpu_demande, resultat, avertissements):
    """Message declenche par le symptome, jamais par l'hypothese de sa cause.

    Un avertissement conditionne par "si un GPU a ete detecte" ne s'affiche pas
    quand la detection echoue, c'est-a-dire precisement quand l'utilisateur a
    besoin d'etre informe. Le symptome observable est ici "le calcul s'est fait
    sur le processeur" ; l'etat des deux couches - runtime du systeme et
    fournisseurs du moteur - est joint au message au lieu de le conditionner.

    Sur un poste sans acceleration possible, un calcul sur processeur est le
    fonctionnement normal : l'ecart n'est signale qu'au regard de l'intention
    exprimee par l'utilisateur, et une seule fois par environnement.
    """
    materiel = materiel_constate(resultat)
    marqueur = lire_marqueur()
    marqueur["dernier_materiel_constate"] = materiel
    if not gpu_demande or materiel != "processeur":
        marqueur["ecart_materiel_signale"] = False
        ecrire_marqueur(marqueur)
        return
    if marqueur.get("ecart_materiel_signale"):
        ecrire_marqueur(marqueur)
        return
    present, detail = sonde_runtime_cuda()
    disponibles = resultat.get("fournisseurs_disponibles") or []
    avertissements.append(
        "Le calcul s'est fait sur le processeur alors que la carte graphique "
        "etait demandee.\n"
        "  Runtime du systeme : %s\n"
        "  Fournisseurs du moteur : %s\n"
        "Le resultat est identique, seulement plus lent. Ce message ne sera "
        "plus affiche pour cet environnement."
        % (detail, ", ".join(disponibles) or "aucun constate"))
    marqueur["ecart_materiel_signale"] = True
    ecrire_marqueur(marqueur)


# ==============================================================================
# 12. CLASSE GREFFON
# ==============================================================================
class IaVisageStudioPlugin(Gimp.PlugIn):

    def do_query_procedures(self):
        journal("do_query_procedures atteint")
        return [PROCEDURE_NAME]

    def do_create_procedure(self, name):
        journal("do_create_procedure: " + str(name))
        proc = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN,
                                       self.run_procedure, None)

        # Chaque appel d'enregistrement est protege individuellement : une
        # exception sur l'un d'eux ferait disparaitre le greffon en entier, des
        # menus comme du navigateur de procedures, sans aucun message.
        enregistrements = [
            ("set_image_types", ("RGB*, GRAY*",)),
            ("set_sensitivity_mask", (Gimp.ProcedureSensitivityMask.DRAWABLE,)),
            ("set_menu_label", ("IA Visage Studio...",)),
            ("add_menu_path", ("<Image>/Filters/IA Suite",)),
            ("set_documentation", (
                "Anonymise, fait sourire, ameliore et colorise, en une passe.",
                "Detecte les visages puis applique les operations cochees dans "
                "un seul sous-processus, en dechargeant chaque modele avant de "
                "charger le suivant.", name)),
            ("set_attribution", ("Suite IA GIMP", "Suite IA GIMP", "2026")),
        ]
        for methode, arguments in enregistrements:
            try:
                getattr(proc, methode)(*arguments)
            except Exception as e:
                journal("enregistrement %s en echec: %s" % (methode, e))

        # Le libelle de chaque option coutue annonce son cout, sans chiffre
        # invente : les tailles sont celles declarees dans la table des
        # modeles, et la taille reelle est affichee avant tout transfert.
        arguments_bool = [
            ("anonymize", "Anonymiser les visages",
             "Flou ou mosaique sur chaque visage detecte. Aucune IA n'est "
             "necessaire pour cette etape. Prioritaire : elle annule le "
             "sourire et l'amelioration.", False),
            ("smile", "Faire sourire les visages",
             "Modifie l'expression (%s)." % annonce_cout(ROLE_SOURIRE), False),
            ("enhance", "Ameliorer les visages",
             "Restauration des details (%s). Sans ce modele, un rehaussement "
             "sans IA est applique." % annonce_cout(ROLE_AMELIORATION), False),
            ("colorize", "Coloriser l'image",
             "Colorise toute l'image en conservant sa luminance (%s)."
             % annonce_cout(ROLE_COLORISATION), False),
            ("allow-download", "Telecharger les modeles manquants",
             "Telechargement automatique sous %s par fichier, taille annoncee "
             "avant de commencer." % octets_lisibles(AUTO_DOWNLOAD_MAX_BYTES),
             True),
            ("use-gpu", "Utiliser la carte graphique",
             "Installe un second environnement Python, nettement plus "
             "volumineux, dedie a la variante GPU. Refuse sans transferer un "
             "octet si le runtime CUDA est absent de ce poste.", False),
            ("reinstall-env", "Reinstaller l'environnement IA",
             "Reconstruit l'environnement Python dedie. Plusieurs centaines de "
             "megaoctets seront telecharges.", False),
        ]
        for nom, court, aide, defaut in arguments_bool:
            try:
                proc.add_boolean_argument(nom, court, aide, defaut,
                                          GObject.ParamFlags.READWRITE)
            except Exception as e:
                journal("argument %s en echec: %s" % (nom, e))

        arguments_int = [
            ("anonymize-mode", "Mode d'anonymisation",
             "0 = flou gaussien, 1 = mosaique", 0, 1, ANONYME_FLOU),
            ("max-faces", "Nombre maximum de visages",
             "Au-dela, les visages les moins surs sont ignores", 1, VISAGES_MAX,
             VISAGES_MAX),
        ]
        for nom, court, aide, mini, maxi, defaut in arguments_int:
            try:
                proc.add_int_argument(nom, court, aide, mini, maxi, defaut,
                                      GObject.ParamFlags.READWRITE)
            except Exception as e:
                journal("argument %s en echec: %s" % (nom, e))

        try:
            proc.add_double_argument(
                "smile-intensity", "Intensite du sourire",
                "Valeur transmise au vecteur d'attributs du modele", 0.0, 1.0,
                1.0, GObject.ParamFlags.READWRITE)
        except Exception as e:
            journal("argument smile-intensity en echec: %s" % e)

        return proc

    def run_procedure(self, proc, run_mode, image, drawables, config, run_data):
        if not drawables or len(drawables) < 1:
            return proc.new_return_values(Gimp.PDBStatusType.CALL_ERROR,
                                          GLib.Error("Aucun calque selectionne."))
        drawable = drawables[0]

        if run_mode == Gimp.RunMode.INTERACTIVE:
            try:
                GimpUi.init(PROCEDURE_NAME)
                dlg = GimpUi.ProcedureDialog.new(proc, config, "IA Visage Studio")
                dlg.fill(None)
                lance = dlg.run()
                dlg.destroy()
                if not lance:
                    return proc.new_return_values(Gimp.PDBStatusType.CANCEL, None)
            except Exception as e:
                journal("dialogue indisponible, execution avec les defauts: %s" % e)

        demandees = []
        if bool(lire_option(config, "anonymize", False)):
            demandees.append("anonymiser")
        if bool(lire_option(config, "smile", False)):
            demandees.append("sourire")
        if bool(lire_option(config, "enhance", False)):
            demandees.append("ameliorer")
        if bool(lire_option(config, "colorize", False)):
            demandees.append("coloriser")

        mode_anonyme = int(lire_option(config, "anonymize-mode", ANONYME_FLOU))
        visages_max = int(lire_option(config, "max-faces", VISAGES_MAX))
        intensite = float(lire_option(config, "smile-intensity", 1.0))
        autoriser_telechargement = bool(lire_option(config, "allow-download", True))
        utiliser_gpu = bool(lire_option(config, "use-gpu", False))
        reinstaller = bool(lire_option(config, "reinstall-env", False))

        if not demandees:
            try:
                Gimp.message(
                    "Aucune operation n'est cochee.\n\n"
                    "Cochez au moins l'une des quatre : anonymiser, faire "
                    "sourire, ameliorer ou coloriser.")
            except Exception:
                pass
            return proc.new_return_values(Gimp.PDBStatusType.CANCEL, None)

        conflits = []
        operations = arbitrer_intentions(demandees, conflits)

        dossier_travail = tempfile.mkdtemp(prefix=PLUGIN_ID + "_")
        avertissements = list(conflits)
        undo_ouvert = False
        archive = None
        resultat = {}
        journal("execution demandee: %s (gpu=%s, telechargement=%s)"
                % (",".join(operations), utiliser_gpu, autoriser_telechargement))

        def progression(texte):
            try:
                Gimp.progress_set_text(texte)
                Gimp.progress_pulse()
            except Exception:
                pass

        try:
            try:
                Gimp.progress_init("Preparation d'IA Visage Studio...")
            except Exception:
                pass

            image.undo_group_start()
            undo_ouvert = True

            # --- Pile technique. L'option GPU est refusee avant tout
            #     telechargement si elle ne peut manifestement pas fonctionner.
            pile = STACK_CPU
            if utiliser_gpu:
                present, detail = sonde_runtime_cuda()
                if present:
                    pile = STACK_GPU
                else:
                    avertissements.append(refus_gpu(detail))
                    journal("option GPU refusee avant tout telechargement: " + detail)
                    utiliser_gpu = False

            python_venv = preparer_environnement(pile, dossier_travail,
                                                 reinstaller, progression)
            env_worker = clean_env()
            fournisseurs = FOURNISSEURS_GPU if utiliser_gpu else FOURNISSEURS_CPU

            roles = roles_necessaires(operations)
            modeles = resoudre_modeles(roles, autoriser_telechargement,
                                       progression, avertissements)

            if utiliser_gpu:
                utiliser_gpu, python_venv, env_worker, fournisseurs = \
                    self._valider_ou_degrader(
                        python_venv, modeles, dossier_travail, progression,
                        avertissements, reinstaller)

            chemin_source = os.path.join(dossier_travail, "source.png")
            progression("Export du calque a traiter...")
            calque_designe = exporter_calque(image, drawable, chemin_source)
            if not calque_designe:
                avertissements.append(
                    "L'image tampon n'a pas pu etre construite : le traitement "
                    "a porte sur le composite visible et non sur le seul calque "
                    "selectionne.")

            active, rx, ry, rl, rh = bornes_selection(image)
            parametres = {
                "image": chemin_source,
                "dossier_sortie": dossier_travail,
                "operations": operations,
                "modeles": {role: modeles.get(role) for role in MODELES},
                "fournisseurs": fournisseurs,
                "roi_visages": [rx, ry, rl, rh] if active else None,
                "pivot_espace": PIVOT_ESPACE,
                "pivot_type": PIVOT_TYPE,
                "pause_entre_modeles_s": PAUSE_ENTRE_MODELES_S,
                "taille_entree_detection": MODELES[ROLE_DETECTION]["taille_entree"],
                "yunet_cote_max": YUNET_COTE_MAX,
                "detection_normalisation": MODELES[ROLE_DETECTION]["normalisation"],
                "detection_nombre_classes": 1,
                "score_min_visage": SCORE_MIN_VISAGE,
                "nms_recouvrement_max": NMS_RECOUVREMENT_MAX,
                "visages_max": visages_max,
                "cote_min_visage_ratio": COTE_MIN_VISAGE_RATIO,
                "marge_visage_ratio": MARGE_VISAGE_RATIO,
                "marge_visage_min_px": MARGE_VISAGE_MIN_PX,
                "fusion_plume_ratio": FUSION_PLUME_RATIO,
                "anonymisation_mode": mode_anonyme,
                "anonyme_flou_ratio": ANONYME_FLOU_RATIO,
                "anonyme_pixels_ratio": ANONYME_PIXELS_RATIO,
                "sourire_taille_entree": MODELES[ROLE_SOURIRE]["taille_entree"],
                "sourire_normalisation": MODELES[ROLE_SOURIRE]["normalisation"],
                "sourire_intensite": intensite,
                "sourire_vecteur_taille": SOURIRE_VECTEUR_TAILLE,
                "sourire_indice_attribut": SOURIRE_INDICE_ATTRIBUT,
                "amelioration_taille_entree":
                    MODELES[ROLE_AMELIORATION]["taille_entree"],
                "amelioration_normalisation":
                    MODELES[ROLE_AMELIORATION]["normalisation"],
                "colorisation_taille_entree":
                    MODELES[ROLE_COLORISATION]["taille_entree"],
                "colorisation_normalisation":
                    MODELES[ROLE_COLORISATION]["normalisation"],
                "rehaussement_force": REHAUSSEMENT_FORCE,
                "rehaussement_rayon": REHAUSSEMENT_RAYON,
                # Le plafond memoire n'est pose que lorsqu'aucun materiel n'est
                # demande : RLIMIT_AS compte l'espace virtuel, dont un contexte
                # CUDA reserve des dizaines de gigaoctets sans les toucher.
                "plafond_memoire_actif": not utiliser_gpu,
                "plafond_memoire_ratio": MEMOIRE_PLAFOND_RATIO,
                "plafond_memoire_octets": 0,
            }
            with open(os.path.join(dossier_travail, "cfg.json"), "w",
                      encoding="utf-8") as f:
                json.dump(parametres, f, indent=2)

            texte_attente = "IA Visage Studio : %s..." % ", ".join(
                LIBELLES_OPERATIONS[op] for op in operations)
            statut, resultat, logs = executer_worker(
                python_venv, dossier_travail, texte_attente, env_worker)

            if statut == "erreur_import":
                # Auto-guerison : l'environnement est incomplet, on le
                # reconstruit et on relance une fois, sans rien demander.
                invalider_marqueur(resultat.get("detail", "import worker"))
                progression("Environnement IA incomplet, reconstruction...")
                python_venv = preparer_environnement(pile, dossier_travail, True,
                                                     progression)
                statut, resultat, logs = executer_worker(
                    python_venv, dossier_travail, texte_attente, env_worker)

            if statut == "annule":
                journal("execution annulee par l'utilisateur")
                return proc.new_return_values(Gimp.PDBStatusType.CANCEL, None)

            # Le succes se mesure a la production effective d'un fichier non
            # vide, jamais a la reussite declaree de l'IA : un repli
            # parfaitement abouti serait sinon traite comme un echec, et le
            # message informant du repli deviendrait inatteignable.
            chemin_resultat = os.path.join(dossier_travail, "resultat.png")
            produit = (os.path.isfile(chemin_resultat)
                       and os.path.getsize(chemin_resultat) > 0)
            if not produit:
                archive = archiver_journaux(
                    dossier_travail, {"operations": operations, "statut": statut})
                raise RuntimeError(self._message_echec(statut, resultat, logs,
                                                       archive))

            for ligne in (resultat.get("degradations") or []):
                if ligne not in avertissements:
                    avertissements.append(ligne)
            if resultat.get("preseance_appliquee_par_worker"):
                # Ne devrait jamais arriver : l'arbitrage a lieu cote greffon.
                # Le signaler est le seul moyen de savoir que ce n'a pas ete
                # le cas.
                journal("ANOMALIE: la preseance a du etre appliquee par le "
                        "worker, l'arbitrage du greffon n'a pas fonctionne")
                avertissements.append(
                    "Anomalie interne : la table de preseance a ete appliquee "
                    "par le worker et non par le greffon. Le resultat est "
                    "correct ; merci de signaler ce message.")

            signaler_ecart_materiel(utiliser_gpu, resultat, avertissements)

            etiquette = etiquette_resultat(resultat)
            calque = charger_calque(image, chemin_resultat)
            try:
                calque.set_name("%s - %s" % (drawable.get_name(), etiquette))
            except Exception:
                pass
            parent = None
            position = 0
            try:
                parent = drawable.get_parent()
            except Exception:
                parent = None
            try:
                position = int(image.get_item_position(drawable))
            except Exception:
                position = 0
            # Position explicite : -1 s'interprete par rapport au calque actif
            # et empile vers le haut, ce qui donne l'ordre inverse de celui
            # qu'on attend des qu'il y a plus d'un calque.
            image.insert_layer(calque, parent, position)
            try:
                calque.set_offsets(0, 0)
            except Exception:
                pass

            ecrire_inventaire({
                "derniere_execution": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dernieres_operations": operations,
                "dernier_etiquetage": etiquette,
                "visages_traites": resultat.get("nombre_visages"),
                "duree_totale_s": resultat.get("duree_totale_s"),
                "materiel_constate": materiel_constate(resultat),
            })

            if avertissements:
                try:
                    Gimp.message("IA Visage Studio - traitement termine avec "
                                 "des reserves :\n\n- "
                                 + "\n\n- ".join(avertissements[:6]))
                except Exception:
                    pass
            journal("succes: %s visage(s), %s"
                    % (resultat.get("nombre_visages"), etiquette))
            return proc.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

        except Exception as e:
            texte = str(e)
            if archive is None and os.path.isdir(dossier_travail):
                archive = archiver_journaux(dossier_travail,
                                            {"operations": operations})
                if archive and "Journaux archives" not in texte:
                    texte += ("\n\nJournaux archives ici (joindre ce dossier a "
                              "tout signalement) :\n  " + archive)
            journal("echec: " + premiere_phrase(texte))
            try:
                Gimp.message("IA Visage Studio - echec :\n\n" + texte)
            except Exception:
                pass
            return proc.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                                          GLib.Error(premiere_phrase(texte)))
        finally:
            # Depiler dans des blocs distincts : si l'un leve, l'autre doit
            # tout de meme s'executer. Et ne depiler que ce qui a ete empile.
            if undo_ouvert:
                try:
                    image.undo_group_end()
                except Exception:
                    pass
            try:
                Gimp.progress_end()
            except Exception:
                pass
            try:
                Gimp.displays_flush()
            except Exception:
                pass
            # Le worker ne survit jamais a son parent, quelle que soit la
            # facon dont on sort d'ici.
            tuer_processus_restants()
            if mode_debug():
                journal("mode mise au point actif, dossier conserve: "
                        + dossier_travail)
            else:
                shutil.rmtree(dossier_travail, ignore_errors=True)

    def _valider_ou_degrader(self, python_venv, modeles, dossier_travail,
                             progression, avertissements, reinstaller):
        """Valide l'acceleration par une inference reelle, ou degrade.

        Interroger la bibliotheque ne prouve rien : onnxruntime repond que
        CUDAExecutionProvider est disponible meme lorsque cuDNN est absent, et
        l'echec ne survient alors qu'au premier noeud de convolution, au milieu
        du travail de l'utilisateur.

        L'acceleration materielle est une optimisation, jamais une
        fonctionnalite : quand elle ne peut pas aboutir, le greffon produit le
        resultat sur le chemin de repli et explique en une phrase pourquoi
        l'autre n'a pas servi. Un message d'erreur a la place d'un calque
        serait un echec de conception.
        """
        # Un echec deja constate ne se reconstate pas. Sans cette memoire, les
        # roues cuDNN etaient retentees et l'inference de controle rejouee a
        # chaque ouverture du filtre, pour aboutir au meme repli.
        connu = verdict_acceleration()
        if connu and connu.get("verdict") == "echec" and not reinstaller:
            journal("acceleration deja ecartee le %s, essai non rejoue"
                    % connu.get("constate_le"))
            avertissements.append(message_acceleration_ecartee(connu, True))
            return (False, preparer_environnement(STACK_CPU, dossier_travail,
                                                  False, progression),
                    clean_env(), FOURNISSEURS_CPU)

        installer_cudnn(python_venv, clean_env(), dossier_travail, progression)
        env_gpu = clean_env(dossiers_dll_paquets(python_venv))

        modele = None
        taille = 64
        for role in (ROLE_DETECTION, ROLE_SOURIRE, ROLE_AMELIORATION,
                     ROLE_COLORISATION, ROLE_DETECTION_YUNET):
            if modeles.get(role):
                modele = modeles[role]
                taille = MODELES[role]["taille_entree"]
                break

        if modele is None:
            # Rien a memoriser : ce n'est pas un verdict sur le materiel, mais
            # l'absence du sujet de l'experience.
            avertissements.append(
                "Acceleration materielle non validee : aucun modele n'etait "
                "disponible pour l'inference de controle. Le traitement se "
                "poursuit sur le processeur.")
            return (False, preparer_environnement(STACK_CPU, dossier_travail,
                                                  False, progression),
                    clean_env(), FOURNISSEURS_CPU)

        constat = valider_acceleration(python_venv, env_gpu, modele, taille,
                                       dossier_travail, FOURNISSEURS_GPU,
                                       progression)
        fournisseur = constat.get("fournisseur") or ""
        reussi = bool(constat.get("ok")) and fournisseur \
            and fournisseur != "CPUExecutionProvider"
        # Le verdict s'ecrit tant que la pile GPU est la pile active : le
        # marqueur en depend.
        memoriser_verdict_acceleration(reussi, constat)
        if reussi:
            journal("acceleration validee par inference reelle: " + fournisseur)
            return True, python_venv, env_gpu, FOURNISSEURS_GPU

        avertissements.append(message_acceleration_ecartee(constat))
        journal("acceleration ecartee: %s | %s"
                % (fournisseur or "aucun fournisseur",
                   str(constat.get("detail"))[:150]))
        return (False, preparer_environnement(STACK_CPU, dossier_travail, False,
                                              progression),
                clean_env(), FOURNISSEURS_CPU)

    def _message_echec(self, statut, resultat, logs, archive):
        explications = {
            "timeout": "le traitement a depasse le delai de %d s et a ete "
                       "interrompu" % DELAI_WORKER_S,
            "annule": "le traitement a ete interrompu a votre demande",
            "erreur_import": "l'environnement IA reste incomplet apres "
                             "reconstruction",
            "erreur_image": "l'image exportee n'a pas pu etre relue",
            "erreur_modele": "un modele ONNX n'a pas pu etre charge",
            "erreur_pivot": "une etape a rendu l'image dans un format "
                            "inattendu ; le traitement a ete arrete avant de "
                            "produire un resultat faux",
            "erreur_inference": "l'inference a echoue",
            "aucun_visage": "aucun visage n'a ete detecte sur cette image ; "
                            "une selection autour du visage aide le detecteur",
            "erreur_ecriture": "le resultat n'a pas pu etre ecrit sur disque",
            "erreur_memoire": "memoire insuffisante pour cette chaine de "
                              "modeles sur cette image",
            "erreur_params": "les parametres transmis au worker etaient "
                             "illisibles",
        }
        lignes = ["RAISON: " + explications.get(
            statut, "le traitement n'a produit aucun fichier (%s)" % statut)]
        detail = (resultat or {}).get("detail")
        if detail:
            lignes.append("")
            lignes.append("Detail technique :")
            lignes.append("  " + str(detail)[:400])
        for note in ((resultat or {}).get("degradations") or [])[:4]:
            lignes.append("  " + str(note)[:200])
        if not detail and logs:
            queue = [l for l in logs.strip().splitlines() if l.strip()][-6:]
            if queue:
                lignes.append("")
                lignes.append("Fin du journal :")
                lignes.extend("  " + l[:200] for l in queue)
        if archive:
            lignes.append("")
            lignes.append("Journaux archives ici (joindre ce dossier a tout "
                          "signalement) :")
            lignes.append("  " + archive)
        return "\n".join(lignes)


Gimp.main(IaVisageStudioPlugin.__gtype__, sys.argv)

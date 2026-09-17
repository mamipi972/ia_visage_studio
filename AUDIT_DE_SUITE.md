# Audit de suite — les cinq greffons ensemble

**Établi le 2026-09-16 par Claude, à la demande de l'auteur du dépôt.**
Portée : les cinq greffons de la suite IA pour GIMP 3.0, lus **dans les
versions réellement installées sur le poste de l'utilisateur** et non dans
celles des dépôts.

---

## Pourquoi ce document

Les conflits de ressources partagées sont invisibles à l'échelle d'un fichier.
Ils n'apparaissent qu'en regardant plusieurs greffons ensemble, et le plus
coûteux d'entre eux — deux greffons qui se réinstallent mutuellement dans le
même environnement virtuel, plusieurs gigaoctets à chaque passage, sans le
moindre message — n'existe qu'à partir du deuxième greffon de la suite.

Cette relecture croisée est une étape de livraison, pas une note de travail.
Elle doit être refaite avant toute nouvelle livraison dans la suite.

## Versions auditées

| Greffon | Version lue | Origine |
| --- | --- | --- |
| `deep_erase` | 2.4 | fichier installé, fourni par l'utilisateur |
| `deep_erase_pro` | 6.12 | fichier installé, fourni par l'utilisateur |
| `ia_detourage` | 3.11.0 | fichier installé, fourni par l'utilisateur |
| `gimp_sam2_segmentation` | 6.3 | fichier installé, fourni par l'utilisateur |
| `ia_visage_studio` | 1.0 | ce dépôt |

> Le dépôt `gimp_sam2_segmentation` contient une 6.4 ; c'est bien la **6.3**
> installée qui a été auditée, parce que c'est elle qui tourne.

---

## Carte des ressources partagées

| Greffon | Pile | Racine des données lourdes | Environnement virtuel | Marqueur d'environnement | Cache d'interpréteur |
| --- | --- | --- | --- | --- | --- |
| `deep_erase` | onnx-lama | **profil GIMP** | `venv-onnx-lama` | `ai_suite_env_onnx-lama.json` | `deep_erase_python_cache.txt` |
| `deep_erase_pro` | torch | **profil GIMP** | `venv-torch` | `ai_suite_env_torch.json` | `ai_suite_python_torch.txt` |
| `ia_detourage` | onnx-cpu / onnx-gpu | **profil GIMP** | `venv-onnx-cpu`, `venv-onnx-gpu` | `ai_suite_env_<pile>.json` | — |
| `gimp_sam2_segmentation` | onnx-cpu | `%LOCALAPPDATA%` | `venv-onnx-cpu` | `env_onnx-cpu.json` | `interpreteur_onnx-cpu.json` |
| `ia_visage_studio` | onnx-cpu / onnx-gpu | `%LOCALAPPDATA%` | `venv-onnx-cpu`, `venv-onnx-gpu` | `env_onnx-cpu_<greffon>.json` | `interpreteur_onnx-cpu.json` |

Les cinq partagent `<profil GIMP>/ai_suite_shared/` pour les fichiers légers
(journaux, marqueurs, caches) et le nom de dossier `ai_suite_shared` pour la
racine des données.

### Modèles

| Greffon | Fichiers | Emplacement |
| --- | --- | --- |
| `deep_erase` | `lama.onnx` | profil GIMP / `models` |
| `deep_erase_pro` | `lama.pt` | profil GIMP / `models` |
| `ia_detourage` | `u2netp.onnx`, `u2net.onnx`, `isnet-general-use.onnx` | profil GIMP / `models` |
| `gimp_sam2_segmentation` | `sam2_hiera_{tiny,small,base_plus,large}.{encoder,decoder}.onnx` | `%LOCALAPPDATA%` / `models` |
| `ia_visage_studio` | `yolov8n-face.onnx`, `attgan_smile.onnx`, `gfpgan_1_4.onnx`, `deoldify_artistic.onnx` | `%LOCALAPPDATA%` / `models` |

**Aucun nom de fichier n'est partagé entre deux greffons.** Les mécanismes de
rangement et de suppression des doublons d'`ia_visage_studio` ne s'appliquent
qu'aux quatre noms de sa propre table, et ne peuvent donc pas toucher au
modèle d'un voisin — c'est vérifié par
`outils/tests_unitaires.py` (« un fichier de taille différente n'est pas
touché »).

---

## Ce qui va bien

**Aucun conflit destructeur.** Deux greffons n'écrivent jamais dans le même
environnement virtuel avec des listes de paquets différentes. Le piège de la
section 10 du scénario — l'aller-retour de réinstallation sans fin — n'existe
pas sur ce poste.

**`ia_visage_studio` et `gimp_sam2_segmentation` partagent réellement leur
environnement.** Même chemin, mêmes paquets aux mêmes bornes
(`numpy>=1.24.0,<3`, `onnxruntime>=1.16.0,<2`,
`opencv-python-headless>=4.8.0,<5`, identiques en 6.3 comme en 6.4). Le premier
installe, le second constate et démarre.

> **Mesuré**, et non déduit d'une lecture du code : sur un venv déjà en place
> et sans marqueur, `ia_visage_studio` retourne l'interpréteur existant,
> lance **zéro** commande `pip` et **zéro** création de venv, puis écrit son
> propre marqueur. Le contrôle est permanent :
> `outils/tests_unitaires.py`, `test_poste_deja_installe`, et il est prouvé
> par mutation — la suppression de l'étape de réparation le fait passer au
> rouge.

**Les marqueurs d'environnement ne se marchent pas dessus.** `env_onnx-cpu.json`
(sam2) et `env_onnx-cpu_ia_visage_studio.json` sont des fichiers distincts. Le
suffixe par greffon d'`ia_visage_studio` est délibéré : le marqueur contient
un numéro de version, et un marqueur commun ferait qu'à chaque bascule entre
deux greffons de versions différentes, chacun jugerait l'environnement périmé.

---

## Écarts constatés

### 1. `venv-onnx-cpu` existe en double, sous deux racines — *non corrigé*

`ia_detourage` construit le sien sous le profil GIMP ; `gimp_sam2_segmentation`
et `ia_visage_studio` sous `%LOCALAPPDATA%`. Le même nom de pile désigne donc
deux contenus différents :

| Emplacement | Contenu |
| --- | --- |
| profil GIMP / `venv-onnx-cpu` | `rembg[cpu]`, `onnxruntime`, `numpy`, `pillow`, `pymatting` |
| `%LOCALAPPDATA%` / `venv-onnx-cpu` | `numpy`, `onnxruntime`, `opencv-python-headless` |

Ce n'est **pas** un conflit : les deux dossiers ne se touchent jamais. C'est une
duplication — `onnxruntime` et `numpy` sont installés deux fois — et un nom qui
ment, puisqu'il annonce une pile commune là où il y en a deux.

**Gravité : faible.** Coût en disque, pas en fonctionnement.
**Où agir :** `ia_detourage` (hors de ce dépôt). Aligner sa racine de données
sur `%LOCALAPPDATA%` ferait converger les deux, à condition que les listes de
paquets soient alors réellement compatibles — ce qui n'est pas le cas
aujourd'hui, `rembg` tirant `pillow` et `pymatting`. Le plus honnête serait de
renommer sa pile (`onnx-rembg`) plutôt que de prétendre la partager.

### 2. Trois greffons sur cinq placent leurs données lourdes dans le profil GIMP — *non corrigé*

`deep_erase`, `deep_erase_pro` et `ia_detourage` construisent leur venv sous
`Gimp.directory()`, c'est-à-dire `AppData\Roaming` sous Windows. La section 21
du scénario l'interdit explicitement : sur un poste d'entreprise à profil
itinérant, ce dossier est synchronisé à chaque ouverture et fermeture de
session. `venv-torch` seul pèse plusieurs gigaoctets.

`gimp_sam2_segmentation` 6.3 et `ia_visage_studio` ont déjà corrigé ce point de
leur côté (`%LOCALAPPDATA%`, `~/.local/share`, `~/Library/Application Support`).

**Gravité : nulle sur un poste personnel, élevée sur un profil itinérant.**
**Où agir :** les trois greffons concernés (hors de ce dépôt). La migration doit
se faire par renommage, jamais par un nouveau téléchargement.

### 3. Les archives de journaux se purgeaient mutuellement — *corrigé ici*

Les cinq greffons écrivent dans `<profil GIMP>/ai_suite_shared/logs` et purgent
tous « tout sauf les N plus récents » sur l'ensemble du dossier. Deux
conventions de nommage y cohabitent :

| Forme | Greffons |
| --- | --- |
| `2026-09-16_19-44-05` | `deep_erase`, `ia_detourage` |
| `20260916-194405-000123` | `gimp_sam2_segmentation`, `ia_visage_studio` |

En ASCII, le tiret (`0x2D`) précède le chiffre (`0x30`). Un tri alphabétique
place donc **toujours** la première forme en tête de liste, et une purge qui
s'y fie supprime systématiquement les archives des greffons qui l'emploient,
quel que soit leur âge. Dix incidents de `gimp_sam2_segmentation` ou
d'`ia_visage_studio` suffisent à effacer tout l'historique d'incidents de
`deep_erase` et d'`ia_detourage`.

Personne ne s'en aperçoit : il n'y a pas d'incident quand un greffon
fonctionne, et c'est précisément le jour où l'on a besoin de ces journaux
qu'on découvre leur absence.

**Corrigé dans `ia_visage_studio` 1.0** : la fonction `archives_du_greffon()`
ne retient que les archives portant un `incident.json` au nom de ce greffon, et
les date par la date du fichier plutôt que par le nom du dossier — de sorte
qu'aucune convention de nommage n'entre plus en jeu. Une archive qu'on ne sait
pas identifier n'est jamais supprimée : la laisser vaut mieux qu'effacer celle
d'un voisin.

Deux contrôles permanents (`test_journaux_et_messages`) et une mutation
(« la purge des journaux redevient alphabétique sur tout le dossier ») le
vérifient.

**Reste à faire :** les quatre autres greffons purgent toujours à l'aveugle.
Tant que l'un d'eux tourne, les archives des autres restent exposées. La
correction est la même dans chacun : écrire un `incident.json` et ne purger que
les siennes.

### 4. Le fichier de confiance des modèles n'est pas mutualisé — *non corrigé*

| Greffon | Fichier |
| --- | --- |
| `deep_erase` | `deep_erase_model_trust.json` |
| `deep_erase_pro` | `ai_suite_model_trust.json` |
| `gimp_sam2_segmentation`, `ia_visage_studio` | `trusted_models.json` |
| `ia_detourage` | aucun |

La section 10 du scénario dit que ce fichier peut rester commun sans risque,
puisqu'il est indexé par nom de fichier. Trois fichiers distincts signifient
qu'un modèle partagé verrait son empreinte mémorisée plusieurs fois, et qu'un
changement d'empreinte serait signalé par un greffon et pas par l'autre.

**Gravité : faible.** Aucun modèle n'est partagé entre greffons aujourd'hui.
**Où agir :** partout, en convergeant vers `trusted_models.json`.

---

## Ce qui a été changé dans ce dépôt

| Changement | Fichier |
| --- | --- |
| Purge des journaux limitée aux archives de ce greffon, datées par leur contenu | `ia_visage_studio.py`, `archives_du_greffon()` |
| `incident.json` écrit **avant** la copie, pour qu'une archive incomplète reste identifiable | `ia_visage_studio.py`, `archiver_journaux()` |
| Contrôle permanent du démarrage sur un venv déjà installé par un voisin | `outils/tests_unitaires.py`, `test_poste_deja_installe` |
| Contrôles permanents : archives d'un voisin intactes, archive non identifiée jamais supprimée | `outils/tests_unitaires.py`, `test_journaux_et_messages` |
| Deux mutations prouvant que ces contrôles savent échouer | `outils/tests_mutation.py` |

---

## Refaire cet audit

Ce que `outils/verifier_livraison.py` vérifie déjà automatiquement, à chaque
livraison : les noms des ressources partagées, le fait que le marqueur porte le
nom du greffon, que le venv n'y porte pas le sien, que les variantes processeur
et GPU vivent dans des environnements distincts, et — si le dépôt voisin est
présent à côté de celui-ci — l'accord avec `gimp_sam2_segmentation` sur
`SHARED_DIR_NAME`, `VARIABLE_DOSSIER`, `VARIABLE_DEBUG` et la liste des
paquets.

Ce qu'il ne peut pas vérifier, et qui demande cette relecture manuelle :

- les greffons dont le dépôt n'est pas présent à côté ;
- la racine des données de chacun (profil GIMP ou emplacement local) ;
- les conventions de nommage des archives de journaux ;
- les noms des fichiers de confiance ;
- les noms de fichiers de modèles, pour s'assurer qu'aucun n'est partagé par
  accident entre deux greffons qui en feraient des usages différents.

Un greffon supplémentaire dans la suite rend cette relecture obligatoire avant
sa livraison.

# Table des valeurs

Toutes les constantes citables du greffon, avec leur nom exact dans le code.
C'est ce document qui rend un écart détectable au moment d'une relecture : la
documentation ne doit jamais citer un seuil, un délai ou une taille qui ne
figure pas ici.

`outils/verifier_livraison.py` relit ce tableau et compare chaque valeur à celle
réellement définie dans `ia_visage_studio.py`. Un chiffre modifié d'un seul côté
fait échouer le contrôle de livraison.

## Mesuré ou déclaré

Une valeur est **mesurée** quand elle provient d'une exécution dont la trace
existe ; **déclarée** quand elle vient d'un choix de conception ou d'une source
externe non vérifiée depuis ce dépôt.

**Aucune constante de ce tableau n'a été mesurée sur un poste GIMP réel.** Ce
dépôt ne dispose ni de GIMP, ni d'aucun des quatre modèles : le réseau de
l'environnement de développement ne joint pas les hébergeurs de poids. Les
tailles de modèles, les adresses de téléchargement et les repères de durée sont
donc **déclarés**, et les cases de mesure en fin de document sont vides plutôt
que remplies d'estimations plausibles.

Ce qui **a** été vérifié depuis ce dépôt, et comment :

| Affirmation | Vérifiée par | Date | Version |
| --- | --- | --- | --- |
| Le greffon se charge et s'exerce hors de GIMP | `outils/tests_unitaires.py`, 109 contrôles | 2026-09-16 | 1.0 |
| Le worker produit les bons artefacts contre une doublure d'`onnxruntime` | `outils/tests_worker.py`, 54 contrôles | 2026-09-16 | 1.0 |
| `run_procedure` aboutit de bout en bout et produit un calque nommé | `outils/tests_integration.py`, 30 contrôles | 2026-09-16 | 1.0 |
| Chaque correctif fait échouer son test quand on le remet en défaut | `outils/tests_mutation.py`, 35 mutations | 2026-09-16 | 1.0 |
| Le plafond mémoire refuse bien une allocation trop grande | `tests_worker.py`, cas 11, POSIX uniquement | 2026-09-16 | 1.0 |
| Les cascades de Haar disparaissent à partir d'OpenCV 5 | constat sur `opencv-python-headless` 5.0.0 et 4.14.0 | 2026-09-16 | 1.0 |
| Le modèle YuNet se télécharge depuis l'adresse déclarée, pèse 232 589 octets, se charge dans OpenCV 4.14 et détecte un visage | `outils/tests_yunet.py`, 14 contrôles sur le **vrai** modèle | 2026-09-17 | 1.0 |
| Les boîtes de YuNet sont bien remises à l'échelle de l'image d'origine | `outils/tests_yunet.py` + 2 mutations | 2026-09-17 | 1.0 |
| Sur un venv déjà installé par un autre greffon : zéro `pip`, zéro création de venv | `outils/tests_unitaires.py`, `test_poste_deja_installe` | 2026-09-16 | 1.0 |
| Les cinq greffons de la suite ne se détruisent pas mutuellement leurs ressources | [`AUDIT_DE_SUITE.md`](AUDIT_DE_SUITE.md), relecture croisée | 2026-09-16 | 1.0 |

## Identité et emplacements

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `PLUGIN_ID` | ia_visage_studio | déclarée | Identifiant du greffon. Il nomme le journal, l'inventaire et le marqueur d'environnement. |
| `PLUGIN_VERSION` | 1.0 | déclarée | Version du greffon, consignée dans le marqueur. Elle vit dans le code, jamais dans le nom du fichier. |
| `PROCEDURE_NAME` | plug-in-ia-visage-studio | déclarée | Nom de la procédure enregistrée auprès de GIMP. |
| `SHARED_DIR_NAME` | ai_suite_shared | déclarée | Dossier partagé par toute la suite. **Commun avec les autres greffons** : le modifier d'un seul côté sépare silencieusement deux greffons qui devaient partager leurs modèles. |
| `VARIABLE_DOSSIER` | GIMP_AI_SUITE_DIR | déclarée | Variable d'environnement facultative qui impose l'emplacement des données volumineuses : autre disque, installation portable, dossier d'entreprise. Jamais nécessaire. |
| `VARIABLE_DEBUG` | GIMP_AI_SUITE_DEBUG | déclarée | Variable d'environnement qui conserve le dossier d'exécution au lieu de le détruire. Outil de développement : l'archivage des journaux, lui, est automatique. |
| `ARCHIVES_A_CONSERVER` | 10 | déclarée | Nombre d'incidents **de ce greffon** conservés sous `logs/` avant purge du plus ancien. Le dossier est partagé par la suite : la purge ne touche qu'aux archives portant un `incident.json` au nom de ce greffon, et les date par le contenu et non par le nom du dossier. |
| `NOM_FICHIER_INCIDENT` | incident.json | déclarée | Fichier déposé dans chaque archive de journaux, qui nomme le greffon, sa version, sa pile et la plateforme. C'est lui qui rend une archive identifiable, donc purgeable par son seul propriétaire. |
| `JOURS_ARCHIVES_ORPHELINES` | 30 | déclarée | Âge au-delà duquel une archive que personne ne revendique peut être supprimée — et seulement si elle ne figure pas non plus parmi les `ARCHIVES_A_CONSERVER` plus récentes d'entre elles. Les deux conditions sont exigées ensemble : un greffon de la suite qui n'aurait pas encore reçu ce correctif garde ainsi ses archives récentes. |
| `API_GIMP_CANDIDATES` | 3.0, 4.0 | déclarée | Versions de l'API GObject essayées, dans l'ordre. Le numéro suit l'API, pas l'application : GIMP 3.0, 3.2 et 3.4 partagent l'API `3.0`. |
| `FICHIER_SOURCES_UTILISATEUR` | sources_modeles.json | déclarée | Fichier facultatif du dossier partagé qui ajoute des adresses de téléchargement. Il existe pour qu'une adresse morte se répare sans toucher au code. |

## Piles techniques et dépendances

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `STACK_CPU` | onnx-cpu | déclarée | Nom de la pile processeur. Il suffixe le venv et le cache d'interpréteur. **Identique à celui du greffon `gimp_sam2_segmentation`, et c'est voulu** : les deux déclarent exactement la même pile et partagent donc `venv-onnx-cpu`. |
| `STACK_GPU` | onnx-gpu | déclarée | Pile GPU, dans un environnement séparé. `onnxruntime` et `onnxruntime-gpu` s'installent dans le même dossier du `site-packages` et se détruisent mutuellement : un environnement par variante est la seule structure saine. |
| `REQUIRED_PACKAGES` | numpy>=1.24.0,<3, opencv-python-headless>=4.8.0,<5, onnxruntime>=1.16.0,<2 | déclarée | Les trois seules dépendances que le worker importe réellement. Ni pillow, ni le paquet contrib d'OpenCV, aucun extra de confort. La borne haute d'OpenCV n'est pas décorative : à partir d'OpenCV 5, le paquet ne livre plus les cascades de Haar du détecteur de repli. |
| `REQUIRED_PACKAGES_GPU` | numpy>=1.24.0,<3, opencv-python-headless>=4.8.0,<5, onnxruntime-gpu>=1.16.0,<2 | déclarée | Même pile, variante GPU. |
| `PAQUETS_CUDNN` | nvidia-cudnn-cu12, nvidia-cudnn-cu11 | déclarée | Roues pip de cuDNN tentées dans cet ordre quand la pile GPU est demandée. Leur échec n'est jamais bloquant. |
| `PY_MIN` | 3.8 | déclarée | Plancher imposé par les roues des dépendances. Un candidat sous ce plancher est écarté. |
| `PY_MAX_TESTED` | 3.12 | déclarée | Dernière version pour laquelle une installation complète a été tentée. **Ce plafond n'interdit rien** : il classe. Sur un poste qui ne possède qu'une version plus récente, cette version est retenue. À relever après chaque campagne de test. |
| `PENALITE_APPLICATION_TIERCE` | 60 | déclarée | Points retirés à un interpréteur livré avec une application tierce (Blender, Inkscape, GIMP...). Déclassement, jamais rejet : sur un poste qui n'a rien d'autre, refuser revient à ne pas fonctionner. |
| `LEGACY_VENV_DIR_NAMES` | venv_onnx, venv | déclarée | Anciens noms d'environnement repris par simple renommage, pour ne jamais imposer un nouveau téléchargement. |

## Délais

Tout processus lancé a un délai maximal, et le dépassement tue le groupe de
processus. Aucune de ces valeurs n'a été mesurée sur un poste réel.

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `DELAI_SONDE_S` | 5 | déclarée | Délai maximal pour tester un candidat interpréteur par exécution. |
| `DELAI_CREATION_VENV_S` | 180 | déclarée | Délai maximal de création de l'environnement virtuel. |
| `DELAI_PIP_S` | 1800 | déclarée | Délai maximal d'installation des dépendances. La pile GPU pèse plusieurs gigaoctets, d'où une valeur nettement supérieure à celle d'un greffon à pile légère. |
| `DELAI_TELECHARGEMENT_S` | 1800 | déclarée | Délai maximal d'un téléchargement de modèle. |
| `DELAI_LECTURE_RESEAU_S` | 30 | déclarée | Délai maximal d'une lecture réseau isolée. |
| `DELAI_VALIDATION_GPU_S` | 180 | déclarée | Délai maximal de l'inférence réelle qui valide l'accélération matérielle. |
| `DELAI_WORKER_S` | 1800 | déclarée | Délai maximal du traitement complet. Il couvre le chargement des modèles, pas seulement le calcul. |

## Modèles

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `AUTO_DOWNLOAD_MAX_BYTES` | 419430400 octets (400 Mo) | déclarée | **Seuil de téléchargement automatique, par fichier.** En deçà, le téléchargement se fait seul mais jamais en silence : la taille réelle annoncée par le serveur est affichée avant de commencer. Au-delà, le greffon refuse, l'opération dégrade et le chemin de dépôt est indiqué. |
| `MODELE_TAILLE_MIN_BYTES` | 1048576 octets (1 Mo) | déclarée | Plancher de vraisemblance. Rejette un fichier nul, tronqué ou remplacé par une page d'erreur, avant toute lecture intégrale. |
| `MODELE_TAILLE_MAX_BYTES` | 2147483648 octets (2 Go) | déclarée | **Plafond de vraisemblance**, sans aucun rapport avec le seuil de téléchargement ci-dessus : au-delà, le fichier déposé n'est manifestement pas un modèle de cette suite. Les deux répondent à deux questions différentes et ne doivent jamais être harmonisés. |
| `ROLES_AVEC_REPLI` | detection, detection_yunet, amelioration | déclarée | Rôles dont l'absence se rattrape par un chemin sans IA. Les deux autres (sourire, colorisation) n'ont pas d'équivalent : leur opération est annulée et le motif est dit en une phrase. |
| `SOURIRE_VECTEUR_TAILLE` | 13 | déclarée | Taille du vecteur d'attributs attendu par le modèle de sourire. Dépend de l'export, jamais devinée par le worker : elle lui est transmise. |
| `SOURIRE_INDICE_ATTRIBUT` | 12 | déclarée | Indice de l'attribut « sourire » dans ce vecteur. |

### Fichiers de modèles

Ces tailles et ces adresses sont **déclarées** : elles n'ont pas pu être
vérifiées depuis ce dépôt. Elles ne servent qu'à l'annonce préalable et au
contrôle d'espace disque — la taille réellement affichée avant un
téléchargement est celle que renvoie le serveur, et le greffon dégrade quand une
adresse ne répond pas.

| Rôle | Fichier attendu | Taille | Entrée | Normalisation | Source |
| --- | --- | --- | --- | --- | --- |
| détection (défaut) | `face_detection_yunet_2023mar.onnx` | 232 589 octets — **mesurée** | variable, `YUNET_COTE_MAX` au plus | gérée par OpenCV | **vérifiée** le 2026-09-17 |
| détection (alternative) | `yolov8n-face.onnx` | 13 Mo | 640 x 640 | `x / 255` | aucune — à fournir |
| sourire | `attgan_smile.onnx` | 150 Mo | 256 x 256 | `x / 127,5 - 1` | aucune — à fournir |
| amélioration | `gfpgan_1_4.onnx` | 340 Mo | 512 x 512 | `x / 127,5 - 1` | déclarée, **non vérifiée** |
| colorisation | `deoldify_artistic.onnx` | 250 Mo | 256 x 256 | `x / 127,5 - 1` | aucune — à fournir |

Les rôles sans adresse déclarée ne sont **jamais** téléchargés automatiquement :
le fichier se dépose à la main dans le dossier des modèles, ou son adresse
s'ajoute dans `sources_modeles.json`. Le libellé de la case de l'interface le
dit **avant** que vous ne la cochiez, et ce libellé est calculé à partir de
cette table — il ne peut donc plus promettre un téléchargement que le code ne
sait pas faire.

Les deux adresses HuggingFace du détecteur YOLOv8 ont été **retirées** le
2026-09-17 : elles renvoyaient `HTTP 401` chez un utilisateur, le dépôt ayant
disparu ou étant devenu privé. Une adresse dont on sait qu'elle ne répond pas
ne vaut pas mieux que pas d'adresse, et elle coûte une requête à chaque
lancement.

La normalisation s'écrit toujours de la même façon dans le code : l'entrée passe
par `x / 255`, puis `(x - moyenne) / ecart`. `(0.5, 0.5)` redonne
`x / 127,5 - 1`, `(0.0, 1.0)` redonne `x / 255`.

## Espace disque

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `DISQUE_REQUIS_VENV_BYTES` | 734003200 octets (700 Mo) | déclarée | Espace exigé avant de créer l'environnement processeur. Le contrôle n'a lieu que lorsqu'une installation est réellement engagée : un environnement déjà complet reste utilisable sur un disque plein. |
| `DISQUE_REQUIS_VENV_GPU_BYTES` | 4294967296 octets (4 Go) | déclarée | Espace exigé avant de créer l'environnement GPU, qui embarque les runtimes CUDA. |
| `DISQUE_MARGE_SECURITE_BYTES` | 314572800 octets (300 Mo) | déclarée | Marge ajoutée à tout besoin avant de conclure que la place manque. |

## Pipeline et format pivot

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `PIVOT_ESPACE` | RGB | déclarée | Espace colorimétrique du format pivot imposé entre deux étapes. |
| `PIVOT_TYPE` | uint8 | déclarée | Type du format pivot. Un oubli de conversion ne lève aucune exception : il produit un visage bleu. Le worker vérifie donc le pivot en sortie de chaque étape. |
| `PRESEANCE_FACIALE` | anonymiser, sourire, ameliorer | déclarée | Table de préséance. L'anonymisation annule et remplace toute autre modification faciale. L'arbitrage a lieu côté greffon, avant l'écriture du fichier de paramètres ; le worker refait le même contrôle et le signale. |
| `PAUSE_ENTRE_MODELES_S` | 0.5 | déclarée | Délai laissé au pilote pour désallouer réellement après la destruction d'une session, avant de charger le modèle suivant. CUDA utilise un allocateur paresseux : sans cette pause, le second modèle peut échouer sur un faux manque de mémoire. |

## Détection des visages

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `YUNET_COTE_MAX` | 1024 | déclarée | Plus grand côté auquel l'image est réduite avant d'être donnée à YuNet, dont le coût croît avec la taille d'entrée. Les boîtes rendues sont ensuite remises à l'échelle de l'image d'origine — conversion couverte par un test géométrique sur le vrai modèle, et par une mutation. |
| `TAILLE_ENTREE_DETECTION` | 640 | déclarée | Côté de l'entrée du détecteur. L'image y est mise en lettre-boîte, et les coordonnées font le chemin inverse ensuite. |
| `SCORE_MIN_VISAGE` | 0.35 | déclarée | Plancher de confiance. Volontairement bas : un visage manquant est invisible — l'utilisateur ne sait pas qu'il aurait dû être traité — alors qu'une zone superflue se corrige en supprimant le calque. |
| `NMS_RECOUVREMENT_MAX` | 0.45 | déclarée | Au-delà de ce recouvrement, deux boîtes désignent le même visage. |
| `VISAGES_MAX` | 32 | déclarée | Nombre maximal de visages traités, et borne haute du réglage de l'interface. |
| `COTE_MIN_VISAGE_RATIO` | 0.01 | déclarée | Côté minimal d'un visage, en fraction du plus petit côté de l'image. En dessous, la vignette envoyée aux modèles ne porte plus assez d'information. Le seuil porte sur le côté et non sur l'aire : un seuil d'aire est une hypothèse sur la forme du sujet en plus d'une hypothèse sur sa taille. |
| `MARGE_VISAGE_RATIO` | 0.35 | déclarée | Marge ajoutée autour de la boîte détectée avant recadrage. Les modèles de restauration attendent le menton et le front, que la boîte de détection coupe. |
| `MARGE_VISAGE_MIN_PX` | 12 | déclarée | Marge minimale en pixels, pour les très petits visages. |

## Anonymisation et fusion

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `ANONYME_FLOU` | 0 | déclarée | Valeur du réglage « Mode d'anonymisation » correspondant au flou gaussien. |
| `ANONYME_PIXELS` | 1 | déclarée | Valeur correspondant à la mosaïque. |
| `ANONYME_FLOU_RATIO` | 0.25 | déclarée | Rayon du flou, en fraction du côté de la vignette. Un rayon fixe laisserait un visage parfaitement lisible sur une photo haute définition. |
| `ANONYME_PIXELS_RATIO` | 0.08 | déclarée | Côté de la mosaïque, en fraction du côté de la vignette. |
| `FUSION_PLUME_RATIO` | 0.12 | déclarée | Largeur du fondu au bord de la vignette réinsérée, en fraction de son côté. Le fondu atteint exactement zéro sur l'anneau extérieur : hors de la zone fondue, l'écart pixel avec l'original est nul, et c'est vérifié en 8 et en 16 bits. |
| `REHAUSSEMENT_FORCE` | 0.6 | déclarée | Force du masque flou du rehaussement sans IA. |
| `REHAUSSEMENT_RAYON` | 3 | déclarée | Rayon, en pixels, de ce même masque flou. |

## Mémoire et accélération

| Constante | Valeur | Nature | Rôle |
| --- | --- | --- | --- |
| `MEMOIRE_PLAFOND_RATIO` | 0.75 | déclarée | Fraction de la mémoire physique utilisée comme plafond `RLIMIT_AS`, **et seulement quand aucun matériel n'est demandé** : un contexte CUDA réserve couramment des dizaines de gigaoctets d'espace virtuel sans les toucher. |
| `FOURNISSEURS_GPU` | CUDAExecutionProvider, ROCMExecutionProvider, DmlExecutionProvider, CoreMLExecutionProvider, CPUExecutionProvider | déclarée | Fournisseurs demandés quand la case « carte graphique » est cochée. La liste est croisée côté worker avec ceux réellement disponibles. |
| `FOURNISSEURS_CPU` | CPUExecutionProvider | déclarée | Fournisseur demandé quand la case est décochée. C'est bien la case qui gouverne le matériel, pas seulement le paquet installé. |

**Aucun plafond mémoire n'est posé sous Windows.** Une protection dont la
correction n'a pas été prouvée par un test vaut moins que son absence, et ce
dépôt ne peut pas exécuter ce test. Le plafond POSIX, lui, est prouvé : le cas
11 de `outils/tests_worker.py` pose une limite basse, demande au processus
d'allouer davantage et lit le résultat dans le code de sortie.

## Marqueurs de diagnostic

Ces marqueurs sont écrits deux fois — une fois dans `MARQUEURS_WORKER`, une fois
dans le texte du worker, qui n'interpole rien. `outils/verifier_livraison.py`
extrait les deux ensembles et les compare.

| Marqueur | Code | Signification |
| --- | --- | --- |
| `[OK_RESULTAT]` | 0 | Traitement terminé, image produite. |
| `[ERR_PARAMS]` | 2 | Fichier de paramètres absent ou illisible. |
| `[ERR_IMPORT]` | 3 | Une dépendance manque dans l'environnement. Le greffon reconstruit et relance une fois, sans rien demander. |
| `[ERR_IMAGE]` | 4 | L'image exportée n'a pas pu être relue. |
| `[ERR_MODELE]` | 5 | Aucun modèle utilisable pour les opérations demandées. |
| `[ERR_PIVOT]` | 6 | Une étape a rendu l'image dans un format inattendu. Le traitement s'arrête **avant** de produire un résultat faux. |
| `[ERR_INFERENCE]` | 7 | Les modèles se chargent, mais le calcul échoue. |
| `[ERR_AUCUN_VISAGE]` | 8 | Aucun visage détecté, et aucune autre opération à appliquer. |
| `[ERR_ECRITURE]` | 9 | Le résultat n'a pas pu être écrit sur disque. |
| `[ERR_MEMOIRE]` | 10 | Mémoire insuffisante pour cette chaîne de modèles sur cette image. |
| `[ERR_INATTENDU]` | 11 | Exception non prévue, avec la queue de la trace. |
| `[INFO_MOTEUR]` | — | Moteur réellement employé, par étape. Valeur constatée par le worker. |
| `[INFO_MATERIEL]` | — | Matériel de calcul réellement employé, par étape. |
| `[INFO_DEGRADATION]` | — | Étape dégradée ou ignorée, avec son motif. |

## Repères de durée

Ces cases sont **vides parce qu'elles n'ont pas été mesurées**. Une estimation
plausible à leur place serait indiscernable d'une mesure, et c'est précisément
ce qu'il faut éviter : sans repère, l'utilisateur ne sait pas distinguer un
traitement en cours d'un blocage, et un repère inventé le trompe.

| Étape | Durée mesurée | Poste | Date |
| --- | --- | --- | --- |
| Première installation (pile processeur) | — | — | — |
| Détection sur une photo 12 Mpx (processeur) | — | — | — |
| Anonymisation de 3 visages (processeur) | — | — | — |
| Sourire sur 1 visage (processeur) | — | — | — |
| Amélioration de 1 visage (processeur) | — | — | — |
| Colorisation d'une photo 12 Mpx (processeur) | — | — | — |
| Gain réel de l'accélération matérielle | — | — | — |

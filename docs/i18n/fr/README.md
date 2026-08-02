# Yanshi RL Lab

[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.2-76b900?style=flat-square)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](../../../LICENSE)
[![Status](https://img.shields.io/badge/Status-v0.1-orange?style=flat-square)](../../../CHANGELOG.md)

<p>
<a href="../../../README.md"><img src="https://img.shields.io/badge/Language-English-2f81f7?style=flat-square" alt="English"></a>
<a href="../zh/README.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-e67e22?style=flat-square" alt="简体中文"></a>
<a href="../ja/README.md"><img src="https://img.shields.io/badge/%E8%A8%80%E8%AA%9E-%E6%97%A5%E6%9C%AC%E8%AA%9E-bf3989?style=flat-square" alt="日本語"></a>
<a href="README.md"><img src="https://img.shields.io/badge/Langue-Fran%C3%A7ais-8250df?style=flat-square" alt="Français"></a>
</p>

Un cadre d'apprentissage par renforcement neutre vis-à-vis des fabricants, pour robots à
pattes : entraîner dans Isaac Lab, valider dans MuJoCo, et noter chaque robot sur la même
épreuve.

<p align="center">
  <img src="../../media/hero-g1.gif" width="31%" alt="Unitree G1 marchant sur sol plat">
  <img src="../../media/hero-x2.gif" width="31%" alt="AgiBot Lingxi X2 en marche">
  <img src="../../media/hero-bhl.gif" width="31%" alt="Berkeley Humanoid Lite en marche">
</p>

---

## Présentation

La plupart des dépôts de locomotion sont bâtis autour d'un seul robot : les particularités du
corps se dispersent dans les recettes de tâches, et passer à un second robot revient à tout
réécrire. Ce dépôt inverse la relation. Un robot est un *profil* — articulations, gains et
correspondance sémantique de ses parties du corps — et les recettes de tâches n'atteignent le
corps qu'à travers ces emplacements sémantiques. Ajouter un fabricant, c'est ajouter un
répertoire, pas modifier le code partagé.

Une politique n'est pas terminée parce que la courbe d'entraînement est belle. Ici elle doit
survivre à un second moteur physique : chaque politique est exportée en ONNX, rejouée dans
MuJoCo par une boucle fermée déterministe, puis jugée face à des seuils figés avant même que
l'exécution n'existe. Ces seuils vivent en YAML, jamais en Python ; en changer un laisse donc
une trace visible et auditable.

Les trois robots de lancement diffèrent assez pour maintenir l'abstraction honnête : un
humanoïde de 1,32 m, un humanoïde de 1,3 m d'un autre fabricant, et un modèle open source
imprimable de 0,6 m, sans articulation de taille et dont les jambes ne font que 40 % de la
longueur.

Le nom est la thèse. Yanshi (偃师) était l'artisan qui présenta au roi Mu de Zhou un automate
capable de marcher — il n'était pas le robot, il était celui qui lui a appris à marcher. Ce
dépôt est l'atelier, pas un corps particulier.

## Yanshi Rank

Chaque robot passe la même épreuve à quatre portes sur sol plat, sous commandes fixes et
inférence déterministe. Les seuils sont préenregistrés robot par robot à partir de l'enveloppe
de commandes sur laquelle il a été entraîné, et une chute fait échouer la porte d'emblée.

| Robot | Rotation sur place | Rotation en marchant | Marche 8 s | Marche lente 8 s | Portes |
|---|---|---|---|---|---|
| Unitree G1 (29 DoF) | 280,4° | r = 0,60 m | 4,30 m | 2,20 m | 4/4 |
| AgiBot Lingxi X2 | 85,5° | r = 2,10 m | 4,58 m | 1,79 m | 4/4 |
| Berkeley Humanoid Lite | 7,1° | r = 1,61 m | 2,75 m | 0,02 m | 2/4 |

Berkeley Humanoid Lite marche, mais seulement dans une zone de confort de commande très
étroite : en dessous de 0,4 m/s il ignore complètement la commande, et les deux portes
échouées se situent hors de cette zone. La cause est identifiée et consignée plutôt que
masquée : l'enveloppe de commandes d'entraînement est bien plus large que les points de
l'épreuve, si bien que les commandes lentes sont rares dans la distribution d'échantillonnage.
Une seule graine aléatoire ; les références multi-graines sont un standard de la v0.2.

## Points clés

- **Le robot est un profil** : tout ce qui n'est vrai que d'un seul corps vit dans `robots/<vendor>/<model>/profile.py`, et un robot au profil complet s'entraîne avec une surcouche de tâche *vide*.
- **Le sim2sim comme critère de fin** : un unique contrat de politique (`schema_version: 2`) porte une exécution de l'entraînement jusqu'au rejeu MuJoCo ; une politique qui ne fonctionne que dans l'entraîneur n'est pas un résultat.
- **Portes déclaratives** : les seuils vivent dans `benchmark/gates/*.yaml`, si bien que déplacer un objectif devient un diff sur un fichier de configuration plutôt qu'une ligne enfouie dans le code de jugement.
- **Ressources par référence** : les modèles de robots tiers sont récupérés depuis des commits amont épinglés, jamais versionnés ici — les licences diffèrent selon le fabricant.
- **Tests exécutables sans GPU** : profils, contrat, analyse des portes et générateur de squelette se testent tous sans simulateur ni GPU.

## Installation

Nécessite [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
et sa dépendance Isaac Sim. Installez cette extension dans le même environnement :

```bash
git clone https://github.com/jeffliulab/yanshi-rl-lab.git
cd yanshi-rl-lab
pip install -e source/yanshi_rl_lab

yanshi doctor          # autodiagnostic de l'environnement ; CPU seul, ne plante jamais
yanshi assets fetch    # récupère les modèles épinglés (absents de ce dépôt)
```

## Démarrage rapide

```bash
# 1. tests -- CPU seul, aucun simulateur requis
pytest -q

# 2. entraînement (via le lanceur d'Isaac Lab, pas python directement)
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0 --headless \
    --num_envs 4096 --max_iterations 10000 --seed 42

# 3. rejouer un checkpoint et enregistrer une vidéo
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0 \
    --checkpoint logs/rsl_rl/<run>/model_9999.pt --video --headless

# 4. portes sim2sim dans MuJoCo -- CPU seul, c'est l'étape de recette
python scripts/sim2sim/run_gates.py \
    --gates benchmark/gates/velocity-flat-turn.yaml \
    --contract logs/rsl_rl/<run>/params/contract.json \
    --policy logs/rsl_rl/<run>/exported/policy.onnx
```

Le code de sortie vaut `0` quand toutes les portes passent, `1` quand l'une échoue, `2` quand
une ligne de veto se déclenche.

## Ajouter un robot

```bash
python scripts/tools/new_robot.py --vendor <vendor> --model <model>
```

Le générateur écrit un squelette de profil et son test. Renseignez articulations, gains et
correspondance sémantique des parties du corps, épinglez la source des ressources dans
`assets/registry.py`, puis entraînez avec une surcouche de tâche vide — n'ajoutez une
surcharge que lorsqu'une mesure en prouve la nécessité, et placez cette mesure dans le message
de commit.

## État actuel

La v0.1 est en cours. Deux des trois robots de lancement passent leurs portes sur sol plat ;
le troisième marche mais manque deux portes pour la raison décrite plus haut. Terrain
accidenté, randomisation de domaine et publication du classement relèvent de la v0.2. Les
résultats présentés ici proviennent d'une seule graine aléatoire.

## Remerciements

Construit sur [Isaac Lab](https://github.com/isaac-sim/IsaacLab) et
[RSL-RL](https://github.com/leggedrobotics/rsl_rl). Les modèles de robots proviennent de leurs
fabricants : [Unitree](https://github.com/unitreerobotics),
[AgiBot](https://github.com/AgibotTech) et
[Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite). Les
conventions destinées aux agents suivent
[agent-rules](https://github.com/jeffliulab/agent-rules).

MIT © 2026 Jeff Liu. Certains scripts de lancement dérivent du gabarit Isaac Lab
(BSD-3-Clause) ; les ressources robotiques restent sous les licences de leurs fabricants —
voir [NOTICE](../../../NOTICE).

**Ce fichier liste les différentes méthodes explorées pour réaliser des apprentissages**

Note sur le bras :
Il semble y avoir des problèmes de connexion avec le bras assez récurents. Des messages types "no signal from device 6" coupent la connexion, peut-être est-ce lié au hard utilisé.

**But :**
- Convertir un déplacement par translation cartésien en consignes pour 6 moteurs. (n'a jamais changé)
- Suggestion (potentiellement compliqué) : Prendre en compte l'orientation de la pince dans l'apprentissage. Je l'ai réalisé en effectuant un test physique : voulant ramasser un objet, la consigne de s'approcher de z=0 a été donnée au bras, et c'est ce qu'il a fait mais la pince était à quelque chose comme 60° par rapport à la verticle. Pour saisir ou lacher un objet, il peut être intéressant d'être à 0° ou 90° par exemple, mais pas 60.

**Position initiale :**
- Position "backhoe" (utilisé pour tous les apprentissages sauf un, la position est disponible dans un maccro de ce nom dans armlearn)
- Position aléatoire changent à chaquea simulation (tenté une fois, l'apprentissage est beaucoup plus long et les bénéfices n'ont pas été explorés. C'est cependant prometteur car cela pourrait aider le bras à apprendre à aller dans certains zones qu'il ne s'autorisait pas)

**Position cible :**
NOTE : Tous les apprentissages ont été effectués avec, grosso modo, des cibles dont les coordonnées x y et z étaient entre 50 et 300. Normalement, l'espace est à présent plus important. Un entraînement prenant des cibles dans tout l'espace disponible semble pouvoir fonctionner mais il aurait besoin de plus de temps et peut être de plus d'actions pas simulation. (plus de 1000)
Pour être sûr que des coordonnées données soient atteignables par le bras, il suffit de se référer au graphique 3D représentant l'espace atteignable (graphique qui peut être retracé avec des données de points aléatoires générés tels que espaceAtteignable.csv).

- Apprentissage sur une cible définie (fonctionne mais évidemment sans généralisation)
- Apprentissage sur plusieurs cibles définies en changeant toutes les x générations (des cibles de plus en plus éloignées du point de départ; fonctionne et permet une légère généralisation fonction du nombre de cibles)
- Apprentissage sur plusieurs cibles définies en changeant toutes les x générations puis après n cibles on passe sur cible aléatoire changeant toutes les x générations (fonctionne relativement bien, approchant à des distances moyennes de 3/4 cm de cibles aléatoires non apprises)
- Même méthode que ci-dessus en passant directement aux cibles aléatoires (fonctionne aussi bien que ci-dessus, ce qui semble indiquer que définir des cibles "à la main" pour faciliter l'apprentissage n'est pas utile)
- Apprentissage sur 100 cibles tirées aléatoirement à chaque génération, les 100 cibles demeurant alors les mêmes pour tous les individus de la génération

**Input :**
- Valeur angulaire des 6 moteurs et position de la cible dans un repère cartésien (utilisé pour quelques apprentissages seulement, l'apprentissage semble long et moins performant ainsi)
- Valeur angulaire des  moteurs et distance entre la cible et la position actuelle du bras selon les 3 directions (utilisé avec presque tous les apprentissages, fonctionne très bien)
- Suggéré, pas testé : Valeur angulaire des moteurs, position de la cible, position du bras
- Suggéré, pas testé : Placer les positions dans un repère cylindrique ou sphérique ?

**Output :**
- 13 valeurs possibles, chacune permettant à un moteur de tourner d'un degré dans un sens ou dans l'autre, ou de ne rien faire (13e valeur) (utilisé tout le temps)
- Suggéré, pas testé : 729 possibilité pour toutes les combinaisons de moteurs possibles (aucun; 0,1,2; 0,5; 0...)
- Suggéré, pas testé : 21 possiblités pour toutes les combinaisons par 2 de moteurs, plus leur activation seule et aucun

**Fitness (score durant l'apprentissage) :**
- Malus proportionnel à la distance entre le bras et la cible à chaque frame plus un malus proportionnel à l'amplitude du mouvement demandé (utilisé au début, semble marcher mais ces deux malus sont difficiles à équilibrer)
- Idem que ci-dessus avec un grand bonus si la distance bras-cible est inférieur à un seuil (3.5 par exemple) (fonctionne mais constitue encore un élément à équilibrer avec les deux autres)
- Malus proportionnel à la distance finale entre le bras et la cible, au lieu d'accumuler un malus à chaque frame (a donné les meilleurs résultats en particulier quand utilisé avec les 100 cibles/génération)

**Méthodes d'évaluation, en chargeant le meilleur tpg manuellement (disponibles dans le resultTester) :**
- Execution des déplacements frame par frame (appuyant sur entrée entre chaque) (runByHand)
- Génération de x (souvent 1000) cibles et affichage de la dernière position du bras à chaque simulation, utilisé pour obtenir des statistiques de résultats (runEvals)
- Execution d'une trajectoire prédéfinie avec le bras "physique". Pour ce faire, on prend chaque point de la trajectoire dans l'ordre et on effectue une simulation dessus. On prend les positions de moteurs d'une frame sur n, (une sur 50 par exemple), et on les ajoute dans une liste de positions. Ensuite, armlearn s'occupe d'effectuer la trajectoire. (runRealArmAuto)
- Deplacement vers des coordonnées spécifiées par l'utilisateur avec le bras "physique". (runRealArmByHand)

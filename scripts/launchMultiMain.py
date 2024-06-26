import json
import os
import subprocess
import shutil

nbSeed = 30
nbconfig = 4
nbCPU = 16

pathFile = os.path.dirname(__file__) + "/"
pathBuild = pathFile + "../build/"

nameTest = "multiTraining"
pathLogs = pathBuild+"outLogs/"
index = 0


for indexConf in range(nbconfig):
    while os.path.exists(pathLogs+nameTest+"_"+str(index)+"/"):
        index += 1

    pathConf = pathLogs+nameTest+"_"+str(index)+"/"

    # Create conf folder
    os.makedirs(pathConf)

    # Create params folder
    pathParams = pathConf + "params/"
    os.makedirs(pathParams)
    # Copy parameter files
    shutil.copy(pathBuild + "params/repoConfig/params_" + str(indexConf) + ".json", pathParams + "params.json")
    shutil.copy(pathBuild + "params/repoConfig/trainParams_" + str(indexConf) + ".json", pathParams + "trainParams.json")
    
    # Load train_params
    with open(f'{pathConf}params/trainParams.json', 'r') as f:
        train_params = json.load(f)

    # Update train_params
    train_params["pathTargetCSV"] = pathBuild + "params/AllTarget.csv"
    train_params["pathValidationTrajectories"] = pathBuild + "params/ValidationTrajectories.txt"
    train_params["loadValidationTrajectories"] = True
    train_params["saveValidationTrajectories"] = False
    # Save train_params
    with open(f'{pathConf}params/trainParams.json', 'w') as f:
        json.dump(train_params, f, indent=4)

    for indexSeed in range(nbSeed):

        # Create seed folders
        pathSeed = pathConf + "seed_" + str(indexSeed) + "/"
        os.makedirs(pathSeed)
        os.makedirs(pathSeed + "/dotfiles")

        print("Start Configuration {} with seed {}".format(indexConf, indexSeed))

        # Run the training
        subprocess.run("./armGegelati {} {}params/".format(indexSeed, pathConf), shell=True)
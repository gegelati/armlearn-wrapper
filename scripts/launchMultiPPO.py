import json
import os
import subprocess
import shutil

nbSeed = 2
nbconfig = 1

pathFile = os.path.dirname(__file__) + "/"

nameTest = "multiPPO"
pathLogs = pathFile + "../build/" + "outLogs/"
pathRootParams = pathFile + "../params/"
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
    shutil.copy(pathRootParams + "repoConfig/params_" + str(indexConf) + ".json", pathParams + "params.json")
    shutil.copy(pathRootParams + "repoConfig/trainParams_" + str(indexConf) + ".json", pathParams + "trainParams.json")
    shutil.copy(pathRootParams + "repoConfig/ppoParams_" + str(indexConf) + ".json", pathParams + "ppoParams.json")
    
    for indexSeed in range(nbSeed):

        # Load train_params
        with open(f'{pathConf}params/trainParams.json', 'r') as f:
            train_params = json.load(f)

        # Update train_params
        train_params["pathTargetCSV"] = pathRootParams + "AllTarget.csv"
        train_params["pathValidationTrajectories"] = pathRootParams + "ValidationTrajectories.txt"
        if(indexSeed > 0):
            train_params["loadValidationTrajectories"] = True
            train_params["saveValidationTrajectories"] = False
        # Save train_params
        with open(f'{pathConf}params/trainParams.json', 'w') as f:
            json.dump(train_params, f, indent=4)

        # Create seed folders
        pathSeed = pathConf + "seed_" + str(indexSeed) + "/"
        os.makedirs(pathSeed)

        pathModel = pathSeed + "model/"
        os.makedirs(pathModel)

        print("Start Configuration {} with seed {}".format(indexConf, indexSeed))

        # Run the training
        subprocess.run("./armPPO {} {}params/ {} {}".format(indexSeed, pathConf, pathModel, pathSeed), shell=True)
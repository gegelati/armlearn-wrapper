import json
import os
import subprocess
import shutil

nbSeed = 0
nbconfig = 0


if(os.path.exists("params")):
    pathBuild = ""
else:
    pathBuild = "/"
pathRepoConfig = pathBuild + "params/repoConfig/"

with open(pathRepoConfig + "launchMultiTraining.txt", "r") as file:
    for line in file.readlines():
        nbSeed = int(line.split()[0])
        nbconfig= int(line.split()[1])

nameTest = "federated"
pathLogs = pathBuild+"outLogs/"+nameTest+"_"
index = 0
while(os.path.exists(pathLogs+str(index)+"/")):
    index += 1

pathLogs = pathLogs+str(index)+"/"
os.mkdir(pathLogs)




for indexConf in range(nbconfig):

    # Create conf folder
    pathConf = pathLogs + "config_" + str(indexConf) + "/"
    os.makedirs(pathConf)
    os.makedirs(pathConf + "params/")
    
    # Copy parameter files
    shutil.copy(pathBuild + "params/repoConfig/params_" + str(indexConf) + ".json", pathConf + "params/params.json")
    shutil.copy(pathBuild + "params/repoConfig/trainParams_" + str(indexConf) + ".json", pathConf + "params/trainParams.json")
    shutil.copy(pathBuild + "params/repoConfig/federatedParams_" + str(indexConf) + ".json", pathConf + "params/federatedParams.json")
    

    for indexSeed in range(nbSeed):

        # Create seed folders
        pathSeed = pathConf + "seed_" + str(indexSeed) + "/"
        os.makedirs(pathSeed)

        # Load data
        with open(f'{pathConf}params/trainParams.json', 'r') as f:
            data = json.load(f)
        # Update data
        data["pathTargetCSV"] = pathBuild + "params/AllTarget.csv"
        data["pathLogs"] = pathSeed
        data["seed"] = indexSeed * data["federatedNbSeed"]
        data["loadValidationTrajectories"] = indexSeed != 0
        data["saveValidationTrajectories"] = indexSeed == 0
        # Save data
        with open(f'{pathConf}params/trainParams.json', 'w') as f:
            json.dump(data, f, indent=4)

        print("Start Configuration {} with seed {}".format(indexConf, indexSeed))

        # Run the federated training
        subprocess.run("./armFederated {}params/".format(pathConf), shell=True)
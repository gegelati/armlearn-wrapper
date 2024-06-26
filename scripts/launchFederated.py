import json
import os
import subprocess
import shutil

nbSeed = 1
nbFederatedSeed = 4
nbconfig = 1

pathFile = os.path.dirname(__file__) + "/"

nameTest = "multiFederated"
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
    pathParamsFed = pathConf + "paramsFedSeed/"
    os.makedirs(pathParamsFed)
    # Copy parameter files
    shutil.copy(pathRootParams + "repoConfig/paramsFed_" + str(indexConf) + ".json", pathParamsFed + "params.json")
    shutil.copy(pathRootParams + "repoConfig/trainParamsFed_" + str(indexConf) + ".json", pathParamsFed + "trainParams.json")
    
    # Load train_params
    with open(f'{pathParamsFed}trainParams.json', 'r') as f:
        train_params = json.load(f)

    # Update train_params
    train_params["pathTargetCSV"] = pathRootParams + "/AllTarget.csv"
    train_params["pathValidationTrajectories"] = pathRootParams + "/ValidationTrajectories.txt"
    train_params["loadValidationTrajectories"] = True
    train_params["saveValidationTrajectories"] = False
    # Save train_params
    with open(f'{pathParamsFed}trainParams.json', 'w') as f:
        json.dump(train_params, f, indent=4)



    # Create params folder
    pathParamsFinal = pathConf + "paramsFinal/"
    os.makedirs(pathParamsFinal)
    # Copy parameter files
    shutil.copy(pathRootParams + "repoConfig/paramsFinal_" + str(indexConf) + ".json", pathParamsFinal + "params.json")
    shutil.copy(pathRootParams + "repoConfig/trainParamsFinal_" + str(indexConf) + ".json", pathParamsFinal + "trainParams.json")
    
    # Load train_params
    with open(f'{pathParamsFinal}trainParams.json', 'r') as f:
        train_params = json.load(f)

    # Update train_params
    train_params["pathTargetCSV"] = pathRootParams + "/AllTarget.csv"
    train_params["pathValidationTrajectories"] = pathRootParams + "/ValidationTrajectories.txt"
    train_params["loadValidationTrajectories"] = True
    train_params["saveValidationTrajectories"] = False
    train_params["startPreviousTPG"] = True
    train_params["namePreviousTPG"] = "out_0000.dot"
    # Save train_params
    with open(f'{pathParamsFinal}trainParams.json', 'w') as f:
        json.dump(train_params, f, indent=4)



    seedUsed = 0
    for indexSeed in range(nbSeed):

        # Create seed folders
        pathSeed = pathConf + "seed_" + str(indexSeed) + "/"
        os.makedirs(pathSeed)

        for indexFedSeed in range(nbFederatedSeed):

            # Create federated seed folders
            pathFedSeed = pathSeed + "FederatedSeed_" + str(indexFedSeed) + "/"
            os.makedirs(pathFedSeed)
            os.makedirs(pathFedSeed + "/dotfiles")


            print("\nStart Configuration {} and seed {} for federated {} using seed {}".format(indexConf, indexSeed, indexFedSeed, seedUsed))

            # Run the training
            subprocess.run("./armGegelati {} {} {}".format(seedUsed,  pathParamsFed, pathFedSeed), shell=True)
            seedUsed+=1
        
        pathFederatedRun = pathSeed + "federatedRun/"
        os.makedirs(pathFederatedRun)
        os.makedirs(pathFederatedRun + "/dotfiles")


        print("\nDo federation of Configuration {} and seed {} using seed {}".format(indexConf, indexSeed, seedUsed))
        subprocess.run("./armFederated {} {} {} {}".format(seedUsed, nbFederatedSeed, pathParamsFed, pathSeed), shell=True)
        seedUsed+=1

        
        print("\nStart Configuration {} and seed {} for federated {} using seed {}".format(indexConf, indexSeed, indexFedSeed, seedUsed))
        subprocess.run("./armGegelati {} {} {}".format(seedUsed, pathParamsFinal, pathFederatedRun), shell=True)
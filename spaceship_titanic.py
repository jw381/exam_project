import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def data_cleaner(path: str) -> pd.DataFrame:
    """Function for importing the data and removing all cases of missing values
    as well as converting data types in preparation for normalisation.

    :param path: Name of the file containing the data
    :return: The cleaned dataset
    """

    print("Cleaning the data")

    # Importing data
    data = pd.read_csv(path)

    # Removing unhelpful columns
    data = data.drop(["Name", "Cabin", "VIP"], axis=1)

    # Filling missing values with the mean
    data["Age"].fillna(data["Age"].mean(), inplace=True)

    # Filling shopping features with 0's
    shopping = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for x in shopping:
        data[x].fillna(value=0, inplace=True)

    # Creating a total spending column
    total = list()
    for x in range(len(data.Spa)):
        total.append(
            int(data.RoomService[x]) +
            int(data.FoodCourt[x]) +
            int(data.ShoppingMall[x]) +
            int(data.Spa[x]) +
            int(data.VRDeck[x]))
    data["Total"] = total

    # Dividing the PassengerId
    grps = list()
    person_grps = list()
    for x in range(len(data.PassengerId)):
        counter = 0
        group = list()
        person_group = list()
        for n in range(len(data.PassengerId[x])):
            if counter < 4:
                group.append(data.PassengerId[x][n])

            elif counter > 4:
                person_group.append(data.PassengerId[x][n])

            counter += 1
        grps.append(group)
        person_grps.append(person_group)

    # Concatenating the values in the separate ID's
    grp_id = list()
    person_id = list()
    for x in range(len(grps)):
        join_group = "".join([str(value) for value in grps[x]])
        join_person = "".join([str(value) for value in person_grps[x]])
        grp_id.append(join_group)
        person_id.append(join_person)
    data["GroupID"] = grp_id
    data["PersonalID"] = person_id

    # Assigning the group size
    data["GroupID"] = data.GroupID.astype(int)
    groups = dict()
    group_key = data.GroupID.value_counts().index[:].tolist()
    group_value = data.GroupID.value_counts().values[:]
    for x in range(len(group_key)):
        groups[group_key[x]] = group_value[x]
    group_size = list()
    for x in range(len(data.GroupID)):
        group_size.append(groups[data.GroupID[x]])
    data["GroupSize"] = group_size

    # Filling HomePlanet missing values with member values
    fill_planets = list()
    for x in range(len(data.HomePlanet)):
        if type(data.HomePlanet[x]) == float:
            if data.GroupID[x] == data.GroupID[x-1] and type(
                    data.HomePlanet[x-1]) != float:
                fill_planets.append(data.HomePlanet[x-1])
            elif data.GroupID[x] == data.GroupID[x+1] and type(
                    data.HomePlanet[x+1]) != float:
                fill_planets.append(data.HomePlanet[x+1])
            else:
                fill_planets.append(np.NaN)
        else:
            fill_planets.append(data.HomePlanet[x])
    data["Home"] = fill_planets
    data = data.drop(["HomePlanet", "GroupID"], axis=1)

    # Filling the remaining Home values
    home = list()
    for x in range(len(data.Home)):
        if type(data.Home[x]) == float:
            if data.Destination[x] == "55 Cancri e":
                home.append("Europa")
            elif data.Destination[x] == "TRAPPIST-1e":
                home.append("Mars")
            else:
                home.append("Earth")
        else:
            home.append(data.Home[x])
    data["Home"] = home

    # Filling the remaining Destination values
    planet = list()
    for x in range(len(data.Destination)):
        if type(data.Destination[x]) == float:
            if data.Home[x] == "Europa":
                planet.append("55 Cancri e")
            elif data.Home[x] == "Earth":
                planet.append("PSO J318.5-22")
            else:
                planet.append("TRAPPIST-1e")
        else:
            planet.append(data.Destination[x])
    data["Destination"] = planet

    # Imputing CryoSleep
    sleep = list()
    for x in range(len(data.CryoSleep)):
        if type(data.CryoSleep[x]) == float:
            if data.Total[x] > 0:
                sleep.append(False)
            else:
                sleep.append(True)
        else:
            sleep.append(data.CryoSleep[x])
    data["CryoSleep"] = sleep

    # Converting CryoSleep to integer
    data.CryoSleep = data.CryoSleep.astype(int)

    # Encoding HomePlanet values
    encoder = OneHotEncoder(dtype=int, sparse=True)
    final = pd.DataFrame(
        encoder.fit_transform(data[["Home"]]).toarray(),
        columns=["Earth", "Europa", "Mars"])
    data = pd.concat([data, final], axis=1, ignore_index=False)
    data = data.drop("Home", axis=1)

    # Encoding Destination values
    encoder = OneHotEncoder(dtype=int, sparse=True)
    final = pd.DataFrame(
        encoder.fit_transform(data[["Destination"]]).toarray(),
        columns=["55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e"])
    data = pd.concat([data, final], axis=1, ignore_index=False)
    data = data.drop("Destination", axis=1)

    # Transforming the target
    if "Transported" in data.columns:
        data["Transported"] = data["Transported"].astype(int)

    print("Done.")
    return data


def normaliser(data: pd.DataFrame) -> pd.DataFrame:
    """The function responsible for normalising the data

    :param data: The DataFrame to normalise
    :return: The normalised DataFrame
    """

    # Normalising Age with StandardScaler
    standard = StandardScaler()
    data["Age"] = standard.fit_transform(data["Age"].values.reshape(-1, 1))

    # Normalising shopping with MaxAbsScaler
    mabs = MaxAbsScaler()
    shopping = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Total"]
    for x in shopping:
        data[x] = mabs.fit_transform(
            data[x].values.reshape(-1, 1))

    # Normalising PersonalID with MinMaxScaler
    minmax = MinMaxScaler()
    data["PersonalID"] = minmax.fit_transform(
        data["PersonalID"].values.reshape(-1, 1))
    data["GroupSize"] = minmax.fit_transform(
        data["GroupSize"].values.reshape(-1, 1))

    return data


if __name__ == "__main__":
    
    # Creating the dataset
    file = "train.csv"
    df = data_cleaner(file)

    # Splitting the dataset
    target = df["Transported"]
    df = df.drop(["Transported", "PassengerId"], axis=1)

    # Normalising the dataset
    df = normaliser(df)

    # Creating a train and test set
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.1, random_state=1)

    # Multi-Layer Perceptron Classifier
    mlpc = MLPClassifier(random_state=1,
                         hidden_layer_sizes=(15, 5),
                         max_iter=10000,
                         tol=0.000001)
    mlpc.fit(X_train, y_train)
    mlpc_y_pred = mlpc.predict(X_test)
    mlpc_result = accuracy_score(y_test, mlpc_y_pred)

    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=1,
                                 n_estimators=1000,
                                 max_depth=7)
    rfc.fit(X_train, y_train)
    rfc_y_pred = rfc.predict(X_test)
    rfc_result = accuracy_score(y_test, rfc_y_pred)

    # Support Vector Classifier
    svc = SVC(kernel="linear")
    svc.fit(X_train, y_train)
    svc_y_pred = svc.predict(X_test)
    svc_result = accuracy_score(y_test, svc_y_pred)

    # Outputting initial results
    print("-"*50)
    print("MLP: {0:.1f}".format(mlpc_result*100))
    print("RFC: {0:.1f}".format(rfc_result*100))
    print("SVC: {0:.1f}".format(svc_result*100))

    results = [mlpc_result, rfc_result, svc_result]
    best = max(results)
    pred_dict = {mlpc_result: mlpc_y_pred, rfc_result: rfc_y_pred, svc_result: svc_y_pred}

    print("-"*50)
    print("Classification report for the most successful model:")
    print(classification_report(y_test, pred_dict[best]))
    print("-"*50)
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred_dict[best]))

    # Validation test data
    print("-"*50)
    print("Test data:")
    file = "test.csv"
    test = data_cleaner(file)

    # Isolating PassengerId for submission
    submission1 = pd.DataFrame(test["PassengerId"])    # Dataframe for mlp
    submission2 = pd.DataFrame(test["PassengerId"])    # Dataframe for rfc
    submission3 = pd.DataFrame(test["PassengerId"])    # Dataframe for svc
    test = test.drop("PassengerId", axis=1)

    # Normalising validation data
    test = normaliser(test)

    # -------------------------------
    # MLPC
    mlpc_test = mlpc.predict(test)
    mlp_final_target = mlpc_test.astype(bool)

    # RFC
    rfc_test = rfc.predict(test)
    rfc_final_target = rfc_test.astype(bool)

    # SVC
    svc_test = svc.predict(test)
    svc_final_target = svc_test.astype(bool)
    # -------------------------------

    # Building the submission dataframe
    submission1["Transported"] = mlp_final_target
    submission2["Transported"] = rfc_final_target
    submission3["Transported"] = svc_final_target

    # Submission to file
    # submission1.to_csv("mlp_submission.csv", index=False)    # File for mlp
    # submission2.to_csv("rfc_submission.csv", index=False)    # File for rfc
    # submission3.to_csv("svc_submission.csv", index=False)    # File for svc


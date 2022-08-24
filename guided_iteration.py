import pathlib
import random
import shutil

import numpy as np
import pandas as pd
from average_decision_ordering import calc_ado, calc_do
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

scaler = preprocessing.StandardScaler()
home = pathlib.Path.cwd()

def random_pairs(x, y, l):
    random.shuffle(x)
    random.shuffle(y)
    rp = np.vstack((x, y)).T
    while len(rp) < l:
        random.shuffle(x)
        random.shuffle(y)
        app_rp = np.vstack((x, y)).T
        rp = np.concatenate((rp, app_rp), axis=0)
    df = pd.DataFrame({"x": rp[:, 0], "y": rp[:, 1]})
    df.drop_duplicates(inplace=True, keep="first")
    return df.to_numpy()

def data_grabber(selected_efps):
    hl = pd.read_parquet(home / "data" / "HL.parquet")
    y = np.load(home / "data" / "targets.npy")
    if len(selected_efps) > 0:
        efps = pd.read_parquet(home / "data" / "EFP.parquet")
        efp_df = efps[selected_efps]
        X = pd.concat([hl, efp_df], axis=1)
    else:
        X = hl.copy()
    X = pd.DataFrame(scaler.fit_transform(X))
    return X, y


def isolate_order(ix, N_pairs):
    if (pass_path / "dif_order.parquet").exists():
        print(f"Skipping isolate_order for pass {ix}")
        dif_data = pd.read_parquet(pass_path / "dif_order.parquet")
        idxp0 = dif_data["idx0"].values
        idxp1 = dif_data["idx1"].values
        return idxp0, idxp1

    # Get the predictions from the previous iteration
    hl_file = pass_path / "test_pred.parquet"
    dfy = pd.read_parquet(hl_file)

    # Separate data into signal and background
    dfy_sb = dfy.groupby("y")

    # Set signal/background
    df0 = dfy_sb.get_group(0)
    df1 = dfy_sb.get_group(1)

    # get the separate sig/bkg indices
    idx0 = df0.index.values.tolist()
    idx1 = df1.index.values.tolist()

    # generate a random set of sig/bkg pairs
    print(f"Generating (N={int(N_pairs):,}) sig/bkg pairs")
    idx_pairs = random_pairs(idx0, idx1, N_pairs)
    print(f"After duplicates remove, (N={int(len(idx_pairs)):,}) remaining")

    idxp0 = idx_pairs[:, 0]
    idxp1 = idx_pairs[:, 1]

    # grab the ll and hl values for those sig/bkg pairs
    dfy0 = dfy.iloc[idxp0]
    dfy1 = dfy.iloc[idxp1]
    ll0 = dfy0["ll"].values
    ll1 = dfy1["ll"].values
    hl0 = dfy0["hl"].values
    hl1 = dfy1["hl"].values

    # find differently ordered pairs
    dos = calc_do(fx0=ll0, fx1=ll1, gx0=hl0, gx1=hl1)

    # let's put all of the data and decision-ordering in 1 data frame
    do_df = pd.DataFrame({
        "idx0": idxp0,
        "idx1": idxp1,
        "ll0": ll0,
        "ll1": ll1,
        "hl0": hl0,
        "hl1": hl1,
        "dos": dos,
    })

    # split the similar and differently ordered sets
    do_df_grp = do_df.groupby("dos")
    dif_df = do_df_grp.get_group(0)
    sim_df = do_df_grp.get_group(1)
    dif_df.reset_index().to_parquet(pass_path / "dif_order.parquet")

    return idxp0, idxp1


def check_efps(ix):
    if (pass_path / "dif_order_ado_comparison.csv").exists():
        print(f"Skipping check_efps for pass {ix}")
        return

    # Load the diff-ordering results
    dif_df = pd.read_parquet(pass_path / "dif_order.parquet")

    # Grab the dif-order indices and ll features corresponding to those
    idx0 = dif_df["idx0"].values
    idx1 = dif_df["idx1"].values
    ll0 = dif_df["ll0"].values
    ll1 = dif_df["ll1"].values
    print(f"Checking ADO on diff-order subset of size N = {len(dif_df):,}")

    # get the efps to check against the dif_order results
    efps = pd.read_parquet(home / "data" / "EFP.parquet")

    # Remove previously selected efps
    for selected_efp in selected_efps:
        print(f"removing efp: {selected_efp}")
        efps.drop(columns=[selected_efp], inplace=True)

    graphs, ados = [], []
    for iy, (graph, efp_df) in enumerate(tqdm(list(efps.items()))):
        # Use the same diff-order sig/bkg pairs to compare with ll predictions
        efp0 = efp_df.iloc[idx0].values
        efp1 = efp_df.iloc[idx1].values

        # Calculate the ado
        ado_val = np.mean(calc_do(fx0=ll0, fx1=ll1, gx0=efp0, gx1=efp1))
        ados.append(ado_val)
        graphs.append(graph)
        # Add results to dataframe
    ado_df = pd.DataFrame({"efp": graphs, "ado": ados})
    ado_df = ado_df.sort_values(by=["ado"], ascending=False)
    ado_df.to_csv(pass_path / "dif_order_ado_comparison.csv")


def get_max_efp(ix):
    df = pd.read_csv(pass_path / "dif_order_ado_comparison.csv", index_col=0)

    # sort by max ado
    dfs = df.sort_values(by=["ado"], ascending=False)
    efp_max = dfs.iloc[0]["efp"]
    ado_max = dfs.iloc[0]["ado"]
    print(f"Maximum dif-order graph selected: {efp_max}")
    return efp_max, ado_max


def train_nn(ix):
    # Find the "first" EFP that is most similar to the NN(LL) predictions
    # Train a simple NN with this first choice
    X, y = data_grabber(selected_efps=selected_efps)

    # Split data as 70% training and 30% test
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.3,
                                                      random_state=42)

    # Further split test into half testing and half validation
    X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                    y_val,
                                                    test_size=0.5,
                                                    random_state=42)
    print(X_train)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    predictions = np.hstack(model.predict(X))
    auc_val = roc_auc_score(y_test, np.hstack(model.predict(X_test)))
    print(f"test-set AUC={auc_val:.4}")
    ll = np.load(home / "data" / "ll_predictions.npy")
    test_df = pd.DataFrame({"hl": predictions, "y": y, "ll": ll})
    test_df.to_parquet(pass_path / "test_pred.parquet")
    return auc_val


# Names to save things under
run_name = "Example"
iteration_path = home / "results" / run_name   
pathlib.Path(iteration_path).mkdir(parents=True, exist_ok=True)

ll_auc = 0.859
selected_efps, aucs, ados = [], [], []

# Starting Conditions
ix, ado_max, auc_val = 0, 0, 0

# Loop for N iterations
iterations = 10
for ix in range(iterations):
    # Define data sub-directories
    pass_path = iteration_path / f"p{ix}"
    model_path = pass_path / "models"
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    # Setting the random seed to a predictable value (in this case iteration index)
    random.seed(ix)

    # Train a NN using current EFP selections (or just HL when ix=0)
    auc_val = train_nn(ix)
    print(f"Iteration {ix} -> AUC: {auc_val:.4}")

    # Store the auc results
    aucs.append(auc_val)
    pass_list = ["hl6"] + selected_efps
    ados_list = [np.nan] + ados
    efp_df = pd.DataFrame({
        "efp": pass_list,
        "auc": aucs,
        "ado": ados_list
    })
    efp_df.to_csv(iteration_path / "selected_efps.csv")

    # Isolate random dif-order pairs
    isolate_order(ix=ix, N_pairs=50_000)

    # Check ado with each EFP for most similar DO on dif-order pairs
    check_efps(ix)

    # Get the max EFP and save it
    efp_max, ado_max = get_max_efp(ix)
    selected_efps.append(efp_max)
    print(f"Selected EFPs in Pass {ix}")
    print(selected_efps)
    ados.append(ado_max)

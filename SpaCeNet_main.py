# %%
import pickle

import numpy as np
import torch
import pandas as pd
import os
import time

from SpaCeNet.model import SpaCeNet
from SpaCeNet.utils import get_device, Theta, unpickle
from SpaCeNet.model import SpaCeNetGridSearch
from argparse import ArgumentParser

"""
docker run -it --rm --gpus \"device=2\" -u $(id -u ${USER}):$(id -g ${USER}) -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 -i SpaCeNet_main.py -en MOSTA30 -dp data -pd MouseBrainAdult_30Percent.npz -na 1 -nb 1 -nr 1 -st True -gs True
docker run -it --rm --gpus \"device=2\" -u $(id -u ${USER}):$(id -g ${USER}) -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 -i SpaCeNet_main.py -en MOSTA30 -dp data -pd MouseBrainAdult_30Percent.npz -ma 0.01 -mb 1e-5 -sr True

docker run -it --rm --gpus \"device=2\" -u $(id -u ${USER}):$(id -g ${USER}) -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 -i SpaCeNet_main.py -en MOSTA10 -dp data -pd MouseBrainAdult_10Percent.npz -na 1 -nb 1 -nr 1 -st True -gs True


docker run -it --rm --gpus \"device=3\" -u $(id -u ${USER}):$(id -g ${USER}) -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 -i SpaCeNet_main.py -en Simulation -dp example -pd simulated_data.pickle -na 5 -nb 5 -nr 6 -rf sim_results/ -ss 1e-6 -e 1e-5 -l 3 -gs True -ar True
docker run -it --rm --gpus \"device=3\" -u $(id -u ${USER}):$(id -g ${USER}) -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 -i SpaCeNet_main.py -en Simulation -dp example -pd simulated_data.pickle -rf sim_results/ -ma 0.1 -mb 0.02 -sr True -l 3 -ar True


docker run -it --rm --gpus \"device=3\" -u $(id -u ${USER}):$(id -g ${USER}) -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 -i SpaCeNet_main.py -en Simulation -dp example -pd seed=4_n=1000_samples=100_phimin=20.00_phimax=20.00.pickle -rf sim_results/ -ma 0.1 -mb 0.02 -sr True -l 3 -ar True

"""


def parse_arguments():
    parser = ArgumentParser(description="ArgParser to run SpaCeNet with data provided by pickle objects")

    ### Required
    parser.add_argument("-en", "--exp_name", dest="exp_name", required=True, help="<required> Name of the experiment, e.g. MOSTA_30")
    parser.add_argument("-dp", "--data_path", dest="data_path", required=True, help="<required> Path to spatial data")
    parser.add_argument("-pd", "--preprocessed_data", dest="preprocessed_data", required=True, help="<required> Filename of processed data (in npz format with)")
    parser.add_argument("-gs", "--grid_search", dest="grid_search", required=False, default=False, type=bool, help="<optional>")
    parser.add_argument("-sr", "--single_run", dest="single_run", required=False, default=False, type=bool, help="<optional>")
    parser.add_argument("-ar", "--analyse_run", dest="analyse_run", required=False, default=False, type=bool, help="<optional>")

    ### Optional
    parser.add_argument("-rf", "--results_folder", dest="results_folder", required=False, default="results/", help="<optional> Path to folder")
    parser.add_argument("-cu", "--cuda", dest="use_cuda", required=False, default=True, type=bool, help="<optional>")
    parser.add_argument("-v", "--verbose", dest="verbose", required=False, default=1, help="<optional>")
    parser.add_argument("-nd", "--normalize_data", dest="normalize_data", required=False, default=True, type=bool, help="<optional>")
    parser.add_argument("-st", "--save_theta", dest="save_theta", required=False, default=False, type=bool, help="<optional>")

    # SpaCeNet Parameters
    parser.add_argument("-eb", "--expansion_base", dest="expansion_base", required=False, default="smoothed_potential", help="<optional>")
    parser.add_argument("-l", "--ExpansionOrder", dest="L", required=False, type=int, default=1, help="<optional>")
    parser.add_argument("-se", "--standardize_expansions", dest="standardize_expansions", required=False, default=False, type=bool, help="<optional>")
    parser.add_argument("-sc", "--scale_correction", dest="scale_correction", required=False, default="min", help="<optional>")
    parser.add_argument("-k", "--NumberNeighbors", dest="k", required=False, type=int, default=5, help="<optional>")
    parser.add_argument("-q", "--quantile", dest="quantile", required=False, type=float, default=0.01, help="<optional>")

    ### Parameter ranges for the grid search
    parser.add_argument("-ma", "--min_alpha", dest="min_alpha", required=False, type=float, default=1e-5, help="<optional>")
    parser.add_argument("-mxa", "--max_alpha", dest="max_alpha", required=False, type=float, default=10, help="<optional>")
    parser.add_argument("-na", "--n_alpha", dest="n_alpha", required=False, type=int, default=5, help="<optional>")
    parser.add_argument("-mb", "--min_beta", dest="min_beta", required=False, type=float, default=1e-5, help="<optional>")
    parser.add_argument("-mxb", "--max_beta", dest="max_beta", required=False, type=float, default=10, help="<optional>")
    parser.add_argument("-nb", "--n_beta", dest="n_beta", required=False, type=int, default=5, help="<optional>")
    parser.add_argument("-nr", "--n_refinements", dest="n_refinements", required=False, type=int, default=6, help="<optional>")
    parser.add_argument("-rc", "--refinement_criterion", dest="refinement_criterion", required=False, default="AIC", help="<optional>")
    parser.add_argument("-mi", "--max_iter", dest="max_iter", required=False, type=float, default=100_000, help="<optional>")
    parser.add_argument("-ss", "--step_size", dest="step_size", required=False, type=float, default=1e-8, help="<optional>")
    parser.add_argument("-e", "--eps", dest="eps", required=False, type=float, default=1e-7, help="<optional>")
    parser.add_argument("-mssr", "--max_step_size_reductions", dest="max_step_size_reductions", required=False, type=int, default=3, help="<optional>")

    return dict(vars(parser.parse_args()))


if __name__ == '__main__':
    args = parse_arguments()
    device = get_device(args["use_cuda"])

    print("#########################")
    print(f"Run " + args["exp_name"])
    print("#########################\n")
    print("Loading preprocessed data...")

    data_type = os.path.splitext(args["preprocessed_data"])[-1]
    if data_type == ".npz":
        data = np.load(args["data_path"] + "/" + args["preprocessed_data"], allow_pickle=True)
    elif data_type == ".pickle" or data_type == ".pkl":
        with open(args["data_path"] + "/" + args["preprocessed_data"], "rb") as f:
            data = pickle.load(f)
    else:
        print("Unknown file type... Can not load data")

    X_mat = torch.tensor(data["X_mat"], device=device, dtype=torch.float64)

    if os.path.exists(args["results_folder"] + args["exp_name"] + "_L" + str(args["L"]) + "theta.pt"):
        print("Load precomputed distance matrix theta...")
        Theta_tens = torch.load(args["results_folder"] + args["exp_name"] + "_theta.pt")
    else:
        print("Generating Theta Matrix...")
        coord_mat = data["coord_mat"]
        theta = Theta(
            coord_mat,
            base=args["expansion_base"],
            L=args["L"],
            scale_correction=args["scale_correction"],
            k=args["k"],
            quantile=args["quantile"],
            standardize=args["standardize_expansions"],
        )
        Theta_tens = torch.tensor(theta, device=device, dtype=torch.float64)
        if args["save_theta"]:
            torch.save(Theta_tens, args["results_folder"] + args["exp_name"] + "_L" + str(args["L"]) + "theta.pt")

    optimizer_options = dict(
        max_iter=args["max_iter"],
        step_size=args["step_size"],
        eps=args["eps"],
        max_step_size_reductions=args["max_step_size_reductions"]
    )

    if args["grid_search"]:
        print("Run grid search:")
        grid_search = SpaCeNetGridSearch(
            min_alpha=args["min_alpha"],
            max_alpha=args["max_alpha"],
            n_alpha=args["n_alpha"],
            min_beta=args["min_beta"],
            max_beta=args["max_beta"],
            n_beta=args["n_beta"],
            n_refinements=args["n_refinements"],
            refinement_criterion=args["refinement_criterion"],
            device=device,
            verbose=args["verbose"],
            optimizer_options=optimizer_options
        )
        results = grid_search.fit(
            X_train=X_mat,
            Theta_train=Theta_tens
        )

        if not os.path.exists(args["results_folder"]):
            os.mkdir(args["results_folder"])

        # save results (and errors if any occurred)
        results.to_csv(args["results_folder"] + args["exp_name"] + "_gridsearch.csv", index=False)

        if grid_search.errors:
            with open(args["results_folder"] + args["exp_name"] + "_gridsearch_error.csv", "w") as error_file:
                for error in grid_search.errors:
                    error_file.write(error + "\n\n\n")

        print("Finished Grid Search!")


    # Fit the best model based on a grid-search or run SpaCeNet with a custom set of parameters (--single_run)
    if args["grid_search"] or args["single_run"]:
        if os.path.exists(args["results_folder"] + args["exp_name"] + "_gridsearch.csv"):
            criterion, opt_func = args["refinement_criterion"], "min"
            if criterion == "BIC":
                opt_func = "max"
            grid_df = pd.read_csv(args["results_folder"] + args["exp_name"] + "_gridsearch.csv")
            best_model = grid_df[grid_df[criterion] == grid_df[criterion].apply(opt_func)]
            alpha, beta, L = best_model[["alpha", "beta", "L"]].to_numpy()[0]
            L = int(L)

            print(f"Run SpaCeNet for the Set of parameters: alpha = {alpha}, beta = {beta}")
        else:
            alpha, beta = args["min_alpha"], args["min_beta"]
            print("No grid_search results available")
            print(f"Use alpha = --min_alpha = {alpha}, beta = --min_beta = {beta}")


        model = SpaCeNet(
            alpha=alpha,
            beta=beta,
            optimizer_options=optimizer_options,
            device=device
        )
        model.fit(X_mat, Theta_tens, reset_params=True)

        print("Save model to: " +args["results_folder"] + args["exp_name"] + "_SpaCeNet_model.pkl")
        model.Theta_train = None
        model.X_train = None
        with open(args["results_folder"] + args["exp_name"] + "_SpaCeNet_model.pkl", "wb") as f:
            pickle.dump(model, f)

        print("Finished running SpaCeNet!")


    # Evaluate best edges and plot standard analysis
    if args["analyse_run"]:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm, Normalize

        print("Analyse trained model:" + args["results_folder"] + args["exp_name"] + "_SpaCeNet_model.pkl")
        with open(args["results_folder"] + args["exp_name"] + "_SpaCeNet_model.pkl", "rb") as f:
            model = unpickle(f, device)
            model.device = device

        if os.path.exists(args["results_folder"] + args["exp_name"] + "_gridsearch.csv"):
            grid_df = pd.read_csv(args["results_folder"] + args["exp_name"] + "_gridsearch.csv")
            print("Plot hyper-parameter surface")
            criterion, opt_func = args["refinement_criterion"], "min"
            if criterion == "BIC":
                opt_func = "max"
            best_model = grid_df[grid_df[criterion] == grid_df[criterion].apply(opt_func)]
            best_alpha, best_beta, L = best_model[["alpha", "beta", "L"]].to_numpy()[0]
            plt.tricontourf(grid_df.alpha, grid_df.beta, grid_df.AIC, 20, cmap="viridis")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim([grid_df.alpha.min(), grid_df.alpha.max()])
            plt.ylim([grid_df.beta.min(), grid_df.beta.max()])
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\beta$")
            plt.colorbar()
            plt.scatter(grid_df.alpha, grid_df.beta, s=2, c="k")
            plt.scatter(best_alpha, best_beta, c="red")
            plt.savefig(args["results_folder"] + args["exp_name"] + "_hyperparam_surface.pdf", bbox_inches="tight")


        print("Plot the estimated Omega and DeltaRho Matrices")
        Omega_mat = model.Omega_mat.detach().cpu().numpy()
        Drho_tens = model.Drho_tens.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 1 + model.L, figsize=(12, 3))
        for l in range(-1, model.L):
            ax = axes[l + 1]
            if l == -1:
                norm = TwoSlopeNorm(vmin=np.min(Omega_mat), vcenter=0, vmax=np.max(Omega_mat))
                ax.set_title(r"$\Omega$")
                img = ax.imshow(Omega_mat, cmap="bwr", norm=norm)
            else:
                norm = TwoSlopeNorm(vmin=np.min(Drho_tens[l]) - 1E-7, vcenter=0, vmax=np.max(Drho_tens[l]) + 1E-7)
                ax.set_title(r"$\Delta\rho^{(" + f"{l + 1})}}$")
                img = ax.imshow(Drho_tens[l], cmap="bwr", norm=norm)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.colorbar(img, ax=ax, location="bottom", pad=0.04, shrink=0.9)
            plt.savefig(args["results_folder"] + args["exp_name"] + "_est_params.pdf")


        print("Save ordered spatial correlations")
        try:
            gene_names = data["GeneNames"][:,1]
        except:
            gene_names = np.arange(1, X_mat.shape[-1]+1, 1)
        df_list = []
        for l in range(model.L):
            param_pair_idx = np.argwhere(np.triu(Drho_tens[l], k=1) != 0)
            temp_df = pd.DataFrame([[gene_names[i], gene_names[j], Drho_tens[l, i, j], l + 1] for i, j in param_pair_idx],
                                   columns=["Gene1", "Gene2", "value", "l"])
            df_list.append(temp_df)

        df_Drho_pairs = pd.concat(df_list)
        df_Drho_pairs.sort_values("value", ascending=False, key=np.abs, inplace=True)
        df_Drho_pairs.to_csv(args["results_folder"] + args["exp_name"] + "_spatial_associations.csv", index=False)
        print("Top Spacial associations")
        print(df_Drho_pairs.head())



        print("Plot pairwise expression")
        coord_mat = data["coord_mat"][0]
        X_mat = data["X_mat"][0]
        for ascending in [True, False]:
            gene1, gene2 = df_Drho_pairs.sort_values("value", ascending=ascending).iloc[0][["Gene1", "Gene2"]]
            idx1, idx2 = np.where(gene_names == gene1), np.where(gene_names == gene2)

            fig, ax = plt.subplots(1, 2, figsize=(5, 5))
            ax[0].scatter(coord_mat[:, 0], coord_mat[:, 1], c=X_mat[:, idx1], s=0.03, cmap='viridis', norm='asinh')
            ax[1].scatter(coord_mat[:, 0], coord_mat[:, 1], c=X_mat[:, idx2], s=0.03, cmap='viridis', norm='asinh')

            ax[0].set_title(gene1)
            ax[1].set_title(gene2)
            ax[0].axis("off")
            ax[1].axis("off")
            plt.savefig(args["results_folder"] + args["exp_name"] + f"_{gene1}_{gene2}.png", dpi=500)







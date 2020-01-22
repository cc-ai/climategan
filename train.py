from comet_ml import Experiment


def recon_criterion(input, target):
    return torch.mean(torch.abs(input - target))


if __name__ == "__main__":

    # ---------------------------
    # ------ Prepare Params -----
    # ---------------------------

    args = parsed_args()

    start_time = time()
    logger.time.start_time = start_time

    opts.output_path = env_to_path(opts.output_path)
    opts.output_path = get_increasable_name(opts.output_path)
    Path(opts.output_path).mkdir()
    print("Running model in", opts.output_path)

    # -----------------------------
    # ----- End of Train Loop -----
    # -----------------------------

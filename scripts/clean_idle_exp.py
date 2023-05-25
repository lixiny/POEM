import shutil
import os


def main():
    exp_root = "exp"
    all_exp_dirs = os.listdir(exp_root)

    for idir in all_exp_dirs:
        need_to_remove = True
        fir = os.path.join(exp_root, idir)
        sub_f = os.listdir(fir)

        if "evaluations" in sub_f:
            eval_f = os.listdir(os.path.join(fir, "evaluations"))
            if len(eval_f) != 0:
                need_to_remove = False

        if "checkpoints" in sub_f:
            ckp_folder = os.path.join(fir, "checkpoints")
            inside_ckp_folder = os.listdir(ckp_folder)
            if len(inside_ckp_folder) > 3:
                need_to_remove = False

        if idir.startswith("debug_"):
            need_to_remove = True

        if need_to_remove is True:
            print(f"remove {fir}")
            input("Confirm ? Press any key to continue, Ctrl+C to exit.")
            shutil.rmtree(fir)


if __name__ == "__main__":
    main()
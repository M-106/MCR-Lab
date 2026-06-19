# -----------
# > Imports <
# -----------
import numpy as np
import matplotlib.pyplot as plt



# ----------------
# > Plot Classes <
# ----------------
class PlotPanel:
    def __init__(self, 
        title:str, 
        image:np.ndarray, 
        cmap:str="viridis",
        vmin:float|None=None,
        vmax:float|None=None
    ):
        self.title = title
        self.x = x
        self.y = y
        self.c = c
        self.s = s
        self.cmap = cmap

class ScatterPanel:
    def __init__(self, 
        title:str, 
        x:np.ndarray, 
        y:np.ndarray, 
        c:np.ndarray, 
        s:float=8, 
        cmap:str="viridis"
    ):
        self.title = title
        self.x = x
        self.y = y
        self.c = c
        self.s = s
        self.cmap = cmap



# -----------------------------
# > Plot Panels and Save them <
# -----------------------------
def plot_and_save_panels(
    panels:list[PlotPanel],
    save_path:str,
    suptitle:str|None=None
):
    fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels), 5))

    if len(panels) == 1:
        axes = [axes]
    
    for ax, panel in zip(axes, panels):
        if isinstance(panel, PlotPanel):
            ax.imshow(
                panel.image,
                cmap=panel.cmap,
                vmin=panel.vmin,
                vmax=panel.vmax
            )
            ax.axis("off")
        elif isinstance(panel, ScatterPanel):
            ax.scatter(
                panel.x,
                panel.y,
                c=panel.c,
                s=panel.s,
                cmap=panel.cmap
            )
            ax.set_aspect("equal")
        else:
            raise ValueError(f"Got Unknown Panel from type: {type(panel)}.")

        ax.set_title(panel.title)

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close(fig)



# -----------------------------
# > Plotting End-to-End Tools <
# -----------------------------
def img_torch_to_numpy(img, take_first_sample_if_batchdim_exist=True):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

        if take_first_sample_if_batchdim_exist and len(img.shape) == 4:
            img = img[0]

        img = np.transpose(1, 2, 0)

    return img

def plot_2d_training_samples(input_img, label_img, pred_img, title, save_path):
    # cast to numpy if not nd array
    input_img = img_torch_to_numpy(input_img)
    label_img = img_torch_to_numpy(label_img)
    pred_img = img_torch_to_numpy(pred_img)

    # create plot content
    panels = [
        PlotPanel("Height", input_img[:, :, 0], "viridis"),
        PlotPanel("Intensity", input_img[:, :, 1], "viridis"),
        PlotPanel("Density", input_img[:, :, 2], "viridis"),
        PlotPanel("GT", label_img[:, :, 0], "viridis"),
        PlotPanel("Prediction", pred_img[:, :, 0], "viridis"),
        PlotPanel("Difference", (label_img != pred_img).astype(np.uint8), "viridis", 0, 1)
    ]

    # plotting & saving
    plot_and_save_panels(panels, save_path, title)

def plot_3d_training_samples(input_pts, label_pts, pred_pts, title, save_path):
    # cast to numpy
    input_pts = (
        input_pts[0].detach().cpu().numpy()
    )  # Shape: [NumPoints, Coordinates]
    label_pts = label_pts[0].detach().cpu().numpy()
    pred_pts = pred_pts[0].detach().cpu().numpy()  # Shape: [NumPoints]

    # create plot content
    panels = [
        ScatterPanel(
            "Intensity",
            input_pts[:, 0],
            input_pts[:, 1],
            input_pts[:, 3],
        ),
        ScatterPanel(
            "Height (Z)",
            input_pts[:, 0],
            input_pts[:, 1],
            input_pts[:, 2],
        )
    ]

    # plotting & saving
    plot_and_save_panels(panels, save_path, title)








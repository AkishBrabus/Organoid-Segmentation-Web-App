
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shiny.types import FileInfo, ImgData
import pandas as pd
from pathlib import Path
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np
import win32ui
from utils.preprocess_images import preprocess_images
from utils.segment import segment
from utils.save_load_utils import *


# Add page title and sidebar
app_ui = ui.page_fluid(
    ui.include_css("style.css"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Segment",
                    ui.input_file("file", "Upload images", accept=[".png", ".jpg", ".jpeg", ".tif", ".tiff"], multiple=True),
                    ui.input_action_button("compute", "Compute", class_="btn-success"),
                ),
                ui.accordion_panel(
                    "Save/Load",
                    ui.download_button("save", "Save", class_="btn-secondary"),
                    ui.input_text("savefile", "", "Experiment"),
                    ui.input_action_button("load", "Load", class_="btn-secondary"),
                    ui.input_file("loadfile", "", accept=[".zip"], multiple=False),
                ),
                id = "sidebar_acc_multiple",
                open = "Segment"
            ),
            open="always"
        ),
        ui.layout_columns(
            ui.card(
                ui.output_plot("organoid_image"),
                ui.input_slider("n","", min=1, max=1, value=1, step=1, width='100%')
            ),
            ui.card(
                ui.output_plot("image_statistics")
            ),
            ui.card(
                "test"
            ),
            ui.card(
                "test"
            ),
            col_widths=[6, 6, 12, 12]
        ),
        #padding=["0px", "100px", "0px", "0px"]
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    
    analyzed_image_list = reactive.value()
    analyzed_image_list_loaded = reactive.value()

    @reactive.effect
    @reactive.event(input.compute)
    def analyzed_image_list_calc():
        file: list[FileInfo] | None = input.file()
        if file is None:
            impaths = None
            imnames = None
        else:
            impaths = [i["datapath"] for i in file]
            imnames = [i["name"] for i in file]
        preprocess_images(impaths, imnames)
        with ui.Progress(min=1, max=len(impaths)+1) as p:
            p.set(message="Creating Branched Erfnet...")
            segment(
                data_dir=os.path.join("temp", "comp", "uploaded"), 
                save_dir=os.path.join("temp", "comp"),
                checkpoint_path=r"training/Colon Organoid 5_2_2024/model/best_iou_model.pth",
                progress_bar = p
                )
        predictions_dir = os.path.join("temp", "comp", "predictions")
        im_dir = os.path.join("temp", "comp", "uploaded", "test", "images")
        seed_dir = os.path.join("temp", "comp", "seeds")
        predlist = [os.path.join(predictions_dir, f) for f in os.listdir(predictions_dir) if os.path.isfile(os.path.join(predictions_dir, f))]
        imlist = [os.path.join(im_dir, f) for f in os.listdir(im_dir) if os.path.isfile(os.path.join(im_dir, f))]
        seedlist =  [os.path.join(seed_dir, f) for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
        analyzed_image_list.set([imlist, predlist, seedlist])

    @reactive.effect
    @reactive.event(analyzed_image_list)
    def analyzed_image_list_load():
        print("test")
        images = [tf.imread(impath) for impath in analyzed_image_list()[0]]
        preds = [tf.imread(predpath) for predpath in analyzed_image_list()[1]]
        seeds = [tf.imread(seedpath) for seedpath in analyzed_image_list()[2]]
        analyzed_image_list_loaded.set([images, preds, seeds])
        
    @reactive.effect
    @reactive.event(analyzed_image_list_loaded)
    def calculate_statistics():
        

    @reactive.effect
    @reactive.event(input.load)
    def load():
        file: list[FileInfo] | None = input.loadfile()
        if file is None:
            fpath = None
        else:
            fpath = file[0]["datapath"]
        load_experiment(fpath)

        predictions_dir = os.path.join("temp", "comp", "predictions")
        im_dir = os.path.join("temp", "comp", "uploaded", "test", "images")
        seed_dir = os.path.join("temp", "comp", "seeds")
        predlist = [os.path.join(predictions_dir, f) for f in os.listdir(predictions_dir) if os.path.isfile(os.path.join(predictions_dir, f))]
        imlist = [os.path.join(im_dir, f) for f in os.listdir(im_dir) if os.path.isfile(os.path.join(im_dir, f))]
        seedlist =  [os.path.join(seed_dir, f) for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
        analyzed_image_list.set([imlist, predlist, seedlist])

    @render.download
    def save():
        if analyzed_image_list() is not None:
            print("test")
            save_fname = save_experiment(input.savefile())
            if save_fname is not None:
                return save_fname

    @reactive.effect
    @reactive.event(analyzed_image_list)
    def adjust_slider():
        imlist = analyzed_image_list()[0]

        #Set the slider to the correct configuration
        max = 1 if imlist is None else len(imlist)
        value = input.n() if input.n()<=max else 1 
        ui.update_slider(
             "n",
             value = value,
             min = 1,
             max = max
        )

    @render.plot()
    def organoid_image():
        imlist = analyzed_image_list_loaded()[0]
        predlist = analyzed_image_list_loaded()[1]
        seedlist = analyzed_image_list_loaded()[2]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        fig.set_frameon(False)
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()
        if imlist is None:
            return fig
        im = imlist[input.n()-1]
        pred = predlist[input.n()-1]
        seed = seedlist[input.n()-1]
        ax1.imshow(im, cmap='gist_gray')
        new_cmp= np.load(r"assets/cmap_60.npy")
        new_cmp = ListedColormap(new_cmp)
        ax2.imshow(pred, cmap=new_cmp, interpolation="None")
        ax3.imshow(seed, cmap='YlGn')
        return fig
    
    @render.plot()
    def image_statistics():
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

    

app = App(app_ui, server)
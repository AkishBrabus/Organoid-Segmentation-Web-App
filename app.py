
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
import cv2
import time
from utils.preprocess_images import preprocess_images
from utils.segment import segment
from utils.save_load_utils import *
from utils.image_analysis_utils import *
from utils.pca_plot import pca_plotter



plt.style.use('seaborn-v0_8-darkgrid')

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
        ui.navset_card_tab(
            ui.nav_panel("Segmentation",
                ui.layout_columns(
                    ui.card(
                        ui.input_selectize("n","", [], width="100%"),
                        ui.output_plot("organoid_image")
                    ),
                    ui.card(
                        ui.output_plot("image_statistics")
                    ),
                    col_widths=[6, 6]
                ),
            ),
            ui.nav_panel("Analysis", 
                ui.layout_columns(
                    ui.card(
                        ui.input_selectize("delimiter","Filename delimiter", {
                            "_": "Underscore _",
                            "-": "Dash -",
                            }, width="100%"),
                        ui.output_data_frame("filename_split")
                    ),
                    ui.card(
                        ui.layout_sidebar(
                            ui.sidebar(
                                ui.input_selectize("pca_plot_label_select", "Labels:", ["test1", "test2"], multiple=True),
                                bg="#f8f8f8",
                            ),  
                            ui.output_plot("pca_plot"),
                        )
                    ),
                    ui.card(
                        ui.output_data_frame("quant_dataframe"),
                        ui.download_button('download_quant_dataframe',"Download"),
                    ),
                    ui.card(
                        ui.output_data_frame("whole_image_quant_dataframe"),
                        ui.download_button('download_whole_image_quant_dataframe',"Download"),
                    ),
                    col_widths=[6, 6]
                ),
            ),
            #ui.nav_panel("C", "Panel C content"),
            id="tab",
        ),
        #padding=["0px", "100px", "0px", "0px"]
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    
    analyzed_image_list = reactive.value()
    analyzed_image_list_loaded = reactive.value()
    quant_df = reactive.value()
    whole_image_quant_df = reactive.value()
    choices = reactive.value()

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

        images = [tf.imread(impath) for impath in analyzed_image_list()[0]]
        preds = [tf.imread(predpath) for predpath in analyzed_image_list()[1]]
        seeds = [tf.imread(seedpath) for seedpath in analyzed_image_list()[2]]
        analyzed_image_list_loaded.set([images, preds, seeds])

        crops = {}
        crop_path = os.path.join("temp", "comp", "crops")
        if not os.path.isdir(crop_path):
            os.makedirs(crop_path)
        recScaleFactor = 1
        with ui.Progress(min=1, max=len(imlist)+1) as p:
            
            for i in range(len(imlist)):
                imname = os.path.splitext(os.path.basename(imlist[i]))[0]
                p.set(i+1, message = "Cropping Images...", detail = "{} ({} of {})".format(imname, i+1, len(imlist)))
                img = images[i]
                pred = preds[i]
                lst, rec = cropToIndividualOrganoids(img, pred, deleteEdgeTouching=True, outSize=128, manualScaleFactor=0.2);
                if rec<recScaleFactor:
                    recScaleFactor = rec
                # print(np.shape(img), set(img.flatten()))
                # plt.imshow(img, cmap=new_cmp, interpolation='None')
                # showImageList([x[0] for x in lst], 8)
                crops[imname] = lst
                # Saves each of the individual organoid images in crops folder
                
                for j in range(len(lst)):
                    org = lst[j]
                    fnameIm = imname+"_im_"+str(j+1)+".tif"
                    fnamePred = imname+"_pred_"+str(j+1)+".tif"

                    cv2.imwrite(os.path.join(crop_path, str(fnameIm)), np.array(org[0]).astype(np.uint8))
                    cv2.imwrite(os.path.join(crop_path, str(fnamePred)), np.array(org[1]).astype(np.float32))



        listDir = os.listdir(os.path.join("temp", "comp", "crops"))
        predDir = []
        imDir = []
        for path in listDir:
            if "pred" in path:
                predDir.append(path)

        predDir.sort()

        data = []
        with ui.Progress(min=1, max=len(predDir)+1) as p:
            i=0
            for predFileName in predDir:
                i+=1
                imname = os.path.basename(predFileName)
                p.set(i, message = "Performing Quantitative Analysis...", detail = "{} ({} of {})".format(imname, i, len(predDir)))
                
                imFileName = predFileName.split('_')
                imFileName[-2] = "im"
                imFileName = '_'.join(imFileName)

                pred = np.array(tf.imread(os.path.join("temp", "comp", "crops", predFileName)))
                im = np.array(tf.imread(os.path.join("temp", "comp", "crops", imFileName)))
                pred = (pred > 0.5) * 1 #binarize
                # plt.imshow(im)

                contours, hierarchy = cv2.findContours(pred, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0]
                # backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                # cv2.drawContours(backtorgb, contours, 1, (0,255,0), 0)
                # plt.imshow(backtorgb)

                orgData = [
                    calcCentroid(pred),
                    calcMinFeret(pred),
                    calcMaxFeret(pred),
                    calcSizeXY(contour),
                    calcAreaAndPerimeter(contour),
                    calcConvexAreaAndPerimeter(contour),
                    calcFormFactorAndRoundness(contour),
                    calcSolidity(contour),
                    calcConvexityDefects(contour),
                    calcEllipse(contour),
                    calcConvexity(contour),
                    calcIntensity(im),
                    calcMeanIntensity(contour, im)
                ]
                #print("orgData", orgData)

                flatOrgData = [imFileName]
                for sublist in orgData:
                    if isinstance(sublist, tuple):
                        for item in sublist:
                            flatOrgData.append(item)
                    else:
                        flatOrgData.append(sublist)

                data.append(flatOrgData)

        df = pd.DataFrame(data, columns=['filename', 'x_centroid', 'y_centroid', 'min_feret', 'max_feret', 'bounding_rect_w', 'bounding_rect_h', 'area', 'perimeter', 'convex_area', 'convex_perimeter', 'form_factor', 'roundness', 'solidity', 'convexity_defects', 'ellipse_x', 'ellipse_y', 'ellipse_major_axis_length', 'ellipse_minor_axis_length', 'ellipse_angle', 'eccentricity', 'convexity', 'intensity', 'mean_intensity'])
        df.to_excel(os.path.join("temp", "comp", "quant.xlsx"))
        quant_df.set(df)
        # print(set(list(pred.flatten())))
        # print(np.max(im), np.min(im))

        wholeImageDict = {}
        for index, row in df.iterrows():
            fileName = row['filename']
            split = fileName.split('_')
            imName = '_'.join(split[:-2])
            if imName not in wholeImageDict.keys():
                wholeImageDict[imName] = []
                wholeImageDict[imName].append(row['area'])
                wholeImageDict[imName].append(1)
            else:
                wholeImageDict[imName][0] += row['area']
                wholeImageDict[imName][1] += 1

        wholeImageData = [[key]+value for key, value in wholeImageDict.items()]
        wholeImageDf = pd.DataFrame(wholeImageData, columns=['filename', 'combined_area', 'organoid_count'])
        wholeImageDf.to_excel(os.path.join("temp", "comp", "quant_combined.xlsx"))
        whole_image_quant_df.set(wholeImageDf)

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

        images = [tf.imread(impath) for impath in analyzed_image_list()[0]]
        preds = [tf.imread(predpath) for predpath in analyzed_image_list()[1]]
        seeds = [tf.imread(seedpath) for seedpath in analyzed_image_list()[2]]
        analyzed_image_list_loaded.set([images, preds, seeds])

        df = pd.read_excel(os.path.join("temp", "comp", "quant.xlsx"), index_col=0)
        wholeImageDf = pd.read_excel(os.path.join("temp", "comp", "quant_combined.xlsx"), index_col=0)
        quant_df.set(df)
        whole_image_quant_df.set(wholeImageDf)

    @render.download
    def save():
        if analyzed_image_list() is not None:
            save_fname = save_experiment(input.savefile())
            if save_fname is not None:
                return save_fname

    @reactive.effect
    @reactive.event(analyzed_image_list_loaded)
    def adjust_slider():
        imlist = analyzed_image_list()[0]

        #Set the slider to the correct configuration
        choics = [os.path.splitext(os.path.basename(impath))[0] for impath in imlist]
        ui.update_selectize(
             "n",
             label = "",
             choices=choics,
             selected = choics[0]
        )
        choices.set(choics)

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
        if input.n() is not None and input.n() != '':
            selected_index = choices().index(input.n())
        else:
            return fig
        im = imlist[selected_index]
        pred = predlist[selected_index]
        seed = seedlist[selected_index]
        ax1.imshow(im, cmap='gist_gray')
        new_cmp= np.load(r"assets/cmap_60.npy")
        new_cmp = ListedColormap(new_cmp)
        ax2.imshow(pred, cmap=new_cmp, interpolation="None")
        ax3.imshow(seed, cmap='YlGn')
        return fig
    
    @render.plot()
    def image_statistics():
        fig, axes = plt.subplots(1,4)
        (ax1, ax2, ax3, ax4) = axes
        for ax in axes:
            ax.yaxis.grid(True)
            ax.set_xticks([])
        
        imlist = analyzed_image_list()[0]
        if input.n() is not None and input.n() != '':
            selected_index = choices().index(input.n())
        else:
            return
        imname = os.path.splitext(os.path.basename(imlist[selected_index]))[0]

        df = quant_df().copy()
        df["filename"] = df["filename"].apply(lambda x: '_'.join(os.path.splitext(x)[0].split('_')[:-2]))
        restricted_df = df[df["filename"]==imname]
        ax1.violinplot(dataset=restricted_df[restricted_df["area"]>0]["area"])
        ax1.set_ylabel("area")
        ax1.set_ylim(0, df["area"].max())
        ax2.violinplot(dataset=restricted_df[restricted_df["eccentricity"]>0]["eccentricity"])
        ax2.set_ylabel("eccentricity")
        ax2.set_ylim(0, 1)
        ax3.violinplot(dataset=restricted_df[restricted_df["convexity_defects"]>0]["convexity_defects"])
        ax3.set_ylabel("convexity defects")
        ax3.set_ylim(0, df["convexity_defects"].max())
        ax4.violinplot(dataset=restricted_df[restricted_df["roundness"]>0]["roundness"])
        ax4.set_ylabel("roundness")
        ax4.set_ylim(0, 1)
        return fig

    @reactive.calc
    def filename_split_df_calc():
        impaths = analyzed_image_list()[0]
        imnames = [os.path.splitext(os.path.basename(impath))[0] for impath in impaths]
        depth = min([len(imname.split(input.delimiter())) for imname in imnames])
        filename_split_list = []
        
        label_num = 0
        for i in range(depth):
            conditions = set()
            for imname in imnames:
                conditions.add(imname.split(input.delimiter())[i])
            if len(conditions) <= 10:
                label_num += 1
                filename_split_list.append(["Label "+str(label_num), str(conditions), i])

        filename_split_df = pd.DataFrame(filename_split_list,  columns =  ["Label", "Conditions", "Depth"])
        return filename_split_df


    @render.data_frame
    def filename_split():
        filename_split_df = filename_split_df_calc()
        return render.DataTable(filename_split_df) 
    
    @render.data_frame
    def quant_dataframe():
        return render.DataGrid(quant_df())
    
    @render.download
    def download_quant_dataframe():
        if quant_df() is not None:
            return os.path.join("temp", "comp", "quant.xlsx")
    
    @render.data_frame
    def whole_image_quant_dataframe():
        return render.DataGrid(whole_image_quant_df())
    
    @render.download
    def download_whole_image_quant_dataframe():
        if whole_image_quant_df() is not None:
            return os.path.join("temp", "comp", "quant_combined.xlsx")

    @reactive.effect
    def update_pca_plot_label_select():
        filename_split_df = filename_split_df_calc()
        choices = filename_split_df["Label"].tolist()
        ui.update_selectize(
            "pca_plot_label_select",
            choices=choices
        )

    @render.plot
    def pca_plot():
        filename_split_df = filename_split_df_calc()
        quant_dataframe = quant_df().copy()
        chosen = list(input.pca_plot_label_select())
        chosen_depths = []
        for i in range(len(chosen)):
            d = filename_split_df.at[filename_split_df.index[filename_split_df['Label']==chosen[i]][0], 'Depth']
            chosen_depths.append(d)

        def get_target_from_filename(filename):
            filename_split = filename.split(input.delimiter())
            target = ""
            if len(chosen_depths)==0:
                target = "all"
            for depth in chosen_depths:
                target = target+filename_split[depth]+" "
            return target

        target = quant_dataframe["filename"].apply(get_target_from_filename).tolist()
        data = quant_dataframe.drop(columns=["filename"])
        fig = pca_plotter(data, target)
        return fig


app = App(app_ui, server)
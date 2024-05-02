
from shiny.express import input, ui


# Add page title and sidebar
ui.page_opts(title="Embedding-Based Segmentation", fillable=True)
with ui.sidebar(open="desktop"):
    ui.input_file("file", "Upload images", accept=[".png", ".jpg", ".jpeg", ".tif", ".tiff"], multiple=True)
    ui.input_action_button("go", "Compute", class_="btn-success")

with ui.layout_columns(col_widths=[6, 6, 12, 12]):
    with ui.card():
        ui.input_slider("n","", min=1, max=30, value=1)
    with ui.card():
        "Card 2"
    with ui.card():
        "Card 3"
    with ui.card():
        "Card 4"




# --------------------------------------------------------
# Reactive calculations and effects
# --------------------------------------------------------


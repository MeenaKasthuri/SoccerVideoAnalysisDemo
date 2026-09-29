import streamlit as st
import os
import json

from main import main

if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

if "results" not in st.session_state:
    st.session_state["results"] = {}

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(

    page_title="Soccer Tactical Analysis",

    layout="wide"

)

# -----------------------------
# Title
# -----------------------------

st.title(
    "Soccer Tactical Analysis Platform"
)
image_path = "images/Streamlit_Image.webp"

if os.path.exists(image_path):
    st.image(
        image_path,
        use_container_width=True
    )

st.write(
    """
    Upload a soccer video and generate:

    - Player tracking
    - Ball tracking
    - 2D pitch visualization
    - Heatmaps
    - Team zones
    - Convex hulls
    - Database exports
    - JSON tracking data
    """
)

# -----------------------------
# Upload Video
# -----------------------------

uploaded_video = st.file_uploader(

    "Upload Soccer Video",

    type=[
        "mp4",
        "avi",
        "mov"
    ]

)


# -----------------------------
# Output Settings
# -----------------------------

st.sidebar.header(
    "Analysis Options"
)

generate_pitch = st.sidebar.checkbox(

    "Generate Pitch View",

    value=True

)

generate_heatmap = st.sidebar.checkbox(

    "Generate Heatmap",

    value=True

)

generate_zones = st.sidebar.checkbox(

    "Generate Team Zones",

    value=True

)

generate_hulls = st.sidebar.checkbox(

    "Generate Hull Overlay",

    value=True

)

generate_overlay_heatmap = st.sidebar.checkbox(

    "Generate Heatmap Overlay",

    value=True

)

export_json = st.sidebar.checkbox(

    "Export JSON",

    value=True

)

export_database = st.sidebar.checkbox(

    "Export Database",

    value=True

)

# -----------------------------
# Output Folder
# -----------------------------

output_folder = st.sidebar.text_input(

    "Output Folder",

    value="output_videos"

)

# -----------------------------
# Video Preview
# -----------------------------

if uploaded_video:
    st.subheader(
        "Input Video"
    )

    st.video(uploaded_video)

# -----------------------------
# Analyze Button
# -----------------------------

if st.button(
        "Analyze Video"
):

    if uploaded_video is None:

        st.warning(
            "Please upload a video first."
        )


    else:

        # Create folders

        os.makedirs(
            "temp",
            exist_ok=True
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        # Save uploaded video

        temp_video = os.path.join(

            "temp",

            uploaded_video.name

        )

        with open(

                temp_video,

                "wb"

        ) as f:

            f.write(

                uploaded_video.read()

            )

        # Create options

        options = {

            "pitch": generate_pitch,

            "heatmap": generate_heatmap,

            "zones": generate_zones,

            "hulls": generate_hulls,

            "overlay_heatmap":
                generate_overlay_heatmap,

            "json":
                export_json,

            "database":
                export_database

        }

        # Run pipeline

        with st.spinner(

                "Analyzing video... This may take a while."

        ):

            results = main(
                temp_video,
                output_folder,
                options
            )

            st.session_state["results"] = results
            st.session_state["analysis_complete"] = True

            ##DEBUG##
            st.write("Results returned:")

            st.write(results)

            st.write(type(results))
            ##END DEBUG##

        st.success(

            "Analysis complete!"

        )

# -----------------------------
# Display Outputs
# -----------------------------

###----TABS----###
if st.session_state.get("analysis_complete", False):
    st.write("Reached tab section")

    results = st.session_state["results"]
    st.write(st.session_state)
    st.write(results)

    st.divider()

    st.header("Analysis Results")

    video_results = {
        title: path
        for title, path in results.items()
        if str(path).endswith(
            (".avi", ".mp4", ".mov")
        )
    }

    st.write(video_results)
    st.write(len(video_results))

    tabs = st.tabs(list(video_results.keys()))

    for tab, (title, path) in zip(
            tabs,
            video_results.items()
    ):

        with tab:
            full_path = os.path.join(output_folder, path)

            st.video(full_path)

            with open(full_path, "rb") as f:
                st.download_button(
                    f"Download {title}",
                    f,
                    file_name=os.path.basename(full_path)
                )
###----END TABS----###

###----EXPORTS----###
if st.session_state.get("analysis_complete", False):

    results = st.session_state["results"]

    st.divider()

    st.header("Exports")

    if "results" in st.session_state:

        results = st.session_state["results"]

        if "JSON" in results:
            st.divider()
            st.header("Tracking JSON")

            json_path = results["JSON"]

            with open(json_path, "r") as f:
                json_data = json.load(f)

            st.subheader("Frame 0")

            st.json(json_data[0])

            st.subheader("Players in Frame 0")

            st.json(json_data[0]["players"])

            with st.expander("View Full JSON"):
                st.json(json_data)

            st.json(json_data)


        with open(json_path, "rb") as f:

            st.download_button(

                "Download JSON",

                data=f,

                file_name="tracking_output.json",

                mime="application/json"

            )

    if "SQLite" in results:
        with open(results["SQLite"], "rb") as f:
            st.download_button(
                "Download SQLite Database",
                f,
                file_name="soccer_tracking.db"
            )

    if "SQL Server" in results:

        st.success("SQL Server updated")
###----END EXPORTS----###
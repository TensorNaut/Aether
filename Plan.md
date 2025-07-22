# `plan.md` for Project AETHER: The Ultimate Moon Dashboard (v2.0)

## 🌟 Mission Statement

To create a comprehensive, interactive, and intelligent web-based dashboard for lunar exploration, centered around the **AETHER** deep learning model. The platform will fuse and analyze multi-modal data from ISRO's Chandrayaan-2 mission to enhance our understanding of the Moon, from its surface composition and shadowed regions to its tenuous plasma environment.

**Core Disciplines:** Data Engineering, Self-Supervised Deep Learning, Geospatial Analysis, Scientific Data Processing, Full-Stack Web Development.

-----

## Phase 0: Foundation & Strategic Setup (Week 1-2)

This phase is about preparing your workshop. A solid foundation prevents major headaches later.

### Step 0.1: Environment & Version Control

  - **What to do:** Set up your project repository and development environment.
  - **How to do it:**
    1.  **GitHub Repo:** Create the `TechieSamosa/Aether` repository. Initialize it with a `README.md`, a `LICENSE` file (MIT), and a Python `.gitignore`.
    2.  **Environment Management:** Use `conda` to create a dedicated environment.
        ```bash
        conda create -n aether python=3.10
        conda activate aether
        ```
    3.  **Initial Libraries:** Install foundational libraries for data handling, geospatial analysis, and NASA's SPICE toolkit.
        ```bash
        pip install jupyterlab numpy pandas matplotlib
        pip install spiceypy astropy rasterio gdal
        ```

### Step 0.2: Data Discovery & Acquisition Plan

  - **What to do:** Understand and download the necessary datasets from the ISRO Science Data Archive (ISDA).
  - **How to do it:**
    1.  **Identify Key Instruments & Data:**
          * **SPICE Kernels:** The absolute priority. These are the ancillary files that provide spacecraft trajectory, orientation, and timing. The manual specifies several types we'll need:
              * **SPK (`.bsp`):** Spacecraft Planet Kernel for ephemeris (position and velocity).
              * **CK (`.bsp`):** Camera-Matrix Kernel for attitude (orientation).
              * **SCLK (`.tsc`):** Spacecraft Clock Kernel for converting between onboard time and Ephemeris Time.
              * **LSK & PCK:** Leapseconds and Planetary Constants kernels.
          * **OHRC:** The primary high-resolution panchromatic camera data to be enhanced. Data is Level-1 (calibrated) and comes in a binary `.img` file with an `.xml` label.
          * **DFSAR:** Dual-Frequency SAR for surface texture and roughness, crucial for context in shadowed regions. Calibrated data (Level-1A/1B) is in GeoTIFF (`.tif`) format.
          * **CLASS:** X-ray Spectrometer for elemental composition mapping. Level-1 data is in FITS (`.fits`) format.
          * **TMC2:** Terrain Mapping Camera for generating Digital Elevation Models (DEMs) from stereo imagery. Derived products (Level-2) are GeoTIFF (`.tif`).
          * **DFRS:** Dual Frequency Radio Science for studying the lunar ionosphere through radio occultation. Level-0 data is in a Raw Data Exchange Format (RDEF) with `.obs`, `.prd`, and `.xml` files.
    2.  **Download Strategy:**
          * Download all **SPICE kernels** for the mission.
          * Select a region of interest, like the **Lunar South Pole**.
          * Download the corresponding **Level-1 Calibrated Data** for OHRC, DFSAR, CLASS, and TMC2. For DFRS, you will need the Level-0 RDEF data for specific occultation events].
    3.  **Organize Data:** Create a robust `data/` directory.
        ```
        data/
        ├── raw_downloads/   # Original downloaded zip files
        ├── processed/       # Model-ready data cubes and profiles
        └── source/          # Unzipped and sorted source data
            ├── ohr/
            ├── dfsar/
            ├── class/
            ├── tmc2/
            ├── dfrs/
            └── spice/       # All SPICE kernels (.bsp, .tsc, etc.)
        ```

-----

## Phase 1: Data Engineering & Preprocessing Pipeline (Week 3-6)

This is the heavy-lifting phase. The goal is an automated pipeline to convert the diverse raw formats into analysis-ready products.

### Step 1.1: The SPICE Engine - Your Geometric Backbone

  - **What to do:** Use `spiceypy` to build a Python module (`spice_utils.py`) that handles all geometric and timing calculations.
  - **How to do it:**
    1.  **Kernel Management:** Write a function that loads all SPICE kernels from your `data/spice/` directory using `spiceypy.furnsh()`.
    2.  **Core Utilities:** Create functions to:
          * Convert UTC strings from filenames to Ephemeris Time (ET).
          * Calculate the position and orientation of the Chandrayaan-2 orbiter at any given ET.
          * Determine the precise location (lat, lon) on the Moon's surface that a specific instrument pixel is viewing. This is the cornerstone of data fusion.

### Step 1.2: Instrument-Specific Data Decoders

  - **What to do:** Build a Python module (`data_loaders.py`) with functions to read the data from each instrument.
  - **How to do it:**
    1.  **OHRC/TMC2 Loader:**
          * Parse the `.xml` label to get image dimensions (`lines`, `elements`) and data type (`UnsignedByte`, `UnsignedLSB2`).
          * Use `numpy.fromfile()` with the correct data type to read the binary `.img` file into a NumPy array. Use `rasterio` to handle this as a geospatial object.
    2.  **CLASS Loader:**
          * Use the `astropy.io.fits` library to open the `.fits` files and extract the spectral data and metadata.
    3.  **DFSAR Loader:**
          * Use `rasterio.open()` to directly read the calibrated GeoTIFF (`.tif`) files.
    4.  **DFRS Loader:**
          * Parse the ASCII `.obs` file to get metadata about the occultation session (start/stop times, frequency, etc.).
          * Read the binary `.prd` file, which contains the raw in-phase (I) and quadrature-phase (Q) signal samples. This will require careful binary file handling according to the RDEF format specification in the manual.

### Step 1.3: The Grand Fusion - Creating Geospatial Data Cubes

  - **What to do:** For the surface-imaging instruments, align and stack their data onto a common grid.
  - **How to do it:**
    1.  **Define a Grid:** Choose a standard projection (e.g., Polar Stereographic for the poles).
    2.  **Reproject Layers:** Using your SPICE utilities and `rasterio`, warp the data from OHRC, DFSAR, CLASS, and the TMC2 DEMs onto this common grid.
    3.  **Stack & Save:** Stack the aligned arrays into multi-channel GeoTIFFs. Each file represents a specific lunar region, and each band represents a different data source. This is your model-ready "data cube."

-----

## Phase 2: AETHER - The Core Deep Learning Model (Week 7-10)

Develop the model to enhance the low-light PSR images.

### Step 2.1: Designing the Self-Supervised Task

  - **What to do:** Formulate a pretext task that forces the model to learn the relationship between different data types.
  - **How to do it:**
    1.  **Masked Image Modeling:** Feed the model the full multi-channel data cube, but with a random patch of the **OHRC channel masked out**.
    2.  **Objective:** The model's goal is to reconstruct the missing OHRC patch by using the other layers as context (e.g., it learns what a surface with a specific radar texture from DFSAR and chemical signature from CLASS *should* look like).

### Step 2.2: Building & Training the Model

  - **What to do:** Implement and train a U-Net-based architecture in PyTorch.
  - **How to do it:**
    1.  **Architecture:** Use a standard U-Net, but modify the input layer to accept your multi-channel data cube.
    2.  **Training:** Write a training script using a data loader that feeds masked cubes to the model. Track experiments using Weights & Biases to log performance and visualize outputs.
    3.  **Output:** The result is a trained model (`aether_v1.pth`) capable of enhancing low-light OHRC images.

-----

## Phase 3: Advanced Analysis & Science Products (Week 11-14)

This phase expands the project's scope beyond a single model into a full-fledged scientific analysis platform.

### Step 3.1: DFRS Data Processing for Ionospheric Profiling

  - **What to do:** Process the DFRS radio occultation data to analyze the Moon's tenuous atmosphere.
  - **How to do it:** Follow the workflow outlined in the DFRS Technical Document:
    1.  **Calculate Observed Doppler:** Process the raw I/Q data from the `.prd` files using a Fast Fourier Transform (FFT) to get the time series of the observed Doppler shift.
    2.  **Calculate Theoretical Doppler:** Use your SPICE engine to compute the expected Doppler shift based purely on the geometry of the spacecraft, Moon, and Earth, without any atmospheric effects.
    3.  **Compute Frequency Residual:** Subtract the theoretical Doppler from the observed Doppler. This residual (`Δf`) is the Doppler shift caused purely by the lunar ionosphere and atmosphere.
    4.  **Derive Profiles:** Use the frequency residual to calculate the Total Electron Content (TEC) and derive altitude profiles of electron density. The output will be scientific data (e.g., CSV files or plots) showing electron density vs. altitude.

### Step 3.2: Automated Crater Detection

  - **What to do:** Use the TMC2 DEMs to build a model that automatically finds and catalogs craters.
  - **How to do it:**
    1.  **Data:** Use the derived DEM GeoTIFFs from TMC2 as input.
    2.  **Model:** Train a U-Net or similar segmentation model to identify the characteristic shape of craters from the elevation data.
    3.  **Output:** A GeoJSON file containing the location, diameter, and depth of every detected crater.

-----

## Phase 4 & 5: Backend, Frontend, & Deployment (Week 15-20)

Build the full-stack application to showcase your work.

### Step 4.1: Backend API with FastAPI

  - **What to do:** Develop a Python API to serve your processed data and model results.
  - **How to do it:** Create the following endpoints:
      * `/tiles/{layer}/{z}/{x}/{y}`: Serves pre-generated map tiles for the various data layers (OHRC, DFSAR, DEM, etc.).
      * `/enhance`: Accepts coordinates, runs the AETHER model on the corresponding data cube, and returns the enhanced image.
      * `/craters`: Returns the GeoJSON of detected craters.
      * `/dfrs/profile/{event_id}`: Returns the calculated electron density profile for a specific DFRS occultation event.

### Step 5.1: Frontend Dashboard with React & Deck.gl

  - **What to do:** Build a highly interactive web interface.
  - **How to do it:**
    1.  **Main View:** A 3D lunar globe that users can explore.
    2.  **Layer Control:** A panel to toggle the visibility of different data layers (base map, enhanced PSRs, crater locations, etc.).
    3.  **AETHER Tool:** An interactive tool to select a PSR and see the model's enhancement in near-real-time.
    4.  **Atmosphere Explorer:** A new section where users can see DFRS occultation paths on the globe. Clicking an event will display the corresponding ionospheric profile chart.

### Step 5.2: Deployment

  - **What to do:** Deploy the application to the cloud using Docker.
  - **How to do it:** Use Google Cloud Run or AWS Elastic Beanstalk for the backend and a static hosting service like Netlify or Firebase Hosting for the frontend.

-----

## Final Phase: Documentation & Presentation (Week 21-22)

  - **What to do:** Document your project thoroughly for your portfolio.
  - **How to do it:**
    1.  **GitHub README:** Finalize the README with a GIF of the dashboard, a link to the live demo, and clear instructions.
    2.  **Project Report/Blog Post:** Write a detailed post explaining your entire journey, from decoding obscure PDS4 formats to training advanced AI models and deploying a full-stack application. This will be your most powerful portfolio piece.

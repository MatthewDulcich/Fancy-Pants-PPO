# Fancy-Pants-PPO
Proximal Policy Optimization applied to the game Fancy Pants

## OS Requirements
- Must run on macOS.
- Developed on MacBooks with M Series chips, running macOS Sequoia 15.1.1.

## Setup Steps
### Step 1: Clone the Repository
- Clone this repository to your local machine.

### Step 2: Installing Dependencies (macOS)
- Use a Conda environment or Python virtual environment (`venv`).
- Install dependencies using `requirements.txt` or `shared_env.yml`:
  ```
  pip install -r requirements.txt
  ```

### Step 3: Download Ruffle Files
- Download the Ruffle files from [Ruffle Downloads](https://ruffle.rs/downloads#website-package).
- Choose the macOS version of the files.
- Unzip and place the `ruffle` folder in your home directory.
- Update the `launch_ruffle.html` file by adding the following script tag to the `<head>` section:
  ```
  <script src="<ruffle directory>/ruffle.js"></script>
  ```

### Step 4: Add Fancy Pants .swf Files
- Place the `.swf` files in the main directory. The structure should look like this:
  ```
  fpaworld3.swf
  ```

### Step 5: Additional Notes
1. **Privacy & Accessibility Settings**
   - Enable **System Settings > Privacy & Security > Accessibility** and toggle the **VSCode** switch on. This allows VSCode to execute files and control the screen.
2. **Safari WebDriver**
   - Enable **Allow Remote Automation** in **Safari > Preferences > Developer** section if controlling Safari via WebDriver.
3. **Display Resolution for 4K MacBooks or Monitors**
   - Adjust the display resolution to 1920x1080 (or the closest ratio):
     - Navigate to **System Settings > Displays > Advanced...**.
     - Select **Show resolutions as list**.
     - Enable **Show all resolutions** and choose **1920x1200**.

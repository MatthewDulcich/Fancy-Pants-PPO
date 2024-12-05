# Fancy-Pants-PPO
Proximal Policy Optimization applied to the game Fancy Pants

## OS Must Run on macOS
Developed on MacBooks with M Series chips, on macOS Sequoia 15.1.1.

## Setup
1. Clone this repository.

## Installing Dependencies (macOS)
### Start a Conda Environment or Python Virtual Environment
- Use `requirements.txt` or `shared_env.yml` to install the necessary packages.

### Install Requirements
```
pip install -r requirements.txt
```

### Download Ruffle Files
We self-host Ruffle files for speed, instead of using the Homebrew version or CDN.

1. Download the Ruffle files from [Ruffle Downloads](https://ruffle.rs/downloads#website-package).
2. Download the macOS version of the files.
3. Unzip the downloaded files and place the `ruffle` folder in your home directory.
4. Update the `launch_ruffle.html` file:
   - Add the following script tag to the `<head>` section:
     ```html
     <script src="<ruffle directory>/ruffle.js"></script>
     ```

### Need Fancy Pants .swf Files
Place the `.swf` files in the main directory. The structure should look like this:
fpaworld3.swf

## Privacy & Accessibility Settings
1. Ensure **System Settings > Privacy & Security > Accessibility > VSCode** switch is toggled on.
   - This allows VSCode to execute files and control the screen.

## Additional Notes
### Safari WebDriver
If controlling Safari via WebDriver:
1. Enable **Allow Remote Automation** in **Safari > Preferences > Developer** section.

### Display Resolution for 4K MacBooks or Monitors
1. Set the display resolution to 1920x1080 (or the closest ratio):
   - Navigate to **System Settings > Displays > Advanced...**.
   - Select **Show resolutions as list**.
   - Check **Show all resolutions** and choose **1920x1200**.
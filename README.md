# Fancy-Pants-PPO
Using Reinforcement Learning, we implemented a custom environment using OpenAI's [Gymnasium](https://gymnasium.farama.org/) and the algorithm Proximal Policy Optimization to play the game Fancy Pants.
We implemented our algorithm on [FPA: World 3.](https://www.bornegames.com/games/fpa-world-3/)

## Software and Hardware Requirements
- Python 3.11
- Must run on macOS
- Developed on MacBooks with M Series chips, running macOS Sequoia 15.1

## Setup Steps
### Step 1: Clone the Repository
  ```
  git clone https://github.com/MatthewDulcich/Fancy-Pants-PPO.git
  ```

### Step 2: Installing and Activating Dependencies (macOS)
- Use a Conda environment or Python virtual environment (`venv`)
- Install dependencies using `requirements.txt` or `shared_env.yml`:
1. `requirements.txt`
  ```
  python3.11 -m venv <environment name>
  ```
  ```
  source <environment name>/bin/activate
  ```
  ```
  pip install -r requirements.txt
  ```
2. `shared_env.yml`:
  ```
  conda env create -f shared_env.yml -n <environment name>
  ```
  ```
  conda activate <environment name>
  ```

### Step 3: Download Ruffle Files
- Download the Ruffle files from [Ruffle Downloads](https://ruffle.rs/downloads#website-package)
- Choose the macOS version of the files
- Unzip and place the `ruffle` folder in your `Fancy-Pants-PPO` directory
- Update the `launch_ruffle.html` file by adding the following script tag to the `<head>` section:
  ```
  <script src="<ruffle directory>/ruffle.js"></script>
  ```

### Step 4: Add Fancy Pants .swf Files
- Place your legally obtained "The Fancy Pants Adventures: World 3" `.swf` file in the `Fancy-Pants-PPO` directory. It should be renamed to `fpaworld3.swf`. The structure should look like this:
  ```
  FANCY-PANTS-PPO
    |- fpaworld3.swf
  ```
- Enter the game using the `python main.py` and enter the main level and then you can `ctrl-c` the file (you should go outside the house)

### Step 5: Additional Setup
1. **Privacy & Accessibility Settings**
   - Enable **System Settings > Privacy & Security > Accessibility** and toggle the **VSCode** switch on. This allows VSCode to execute files and control the screen
2. **Safari WebDriver (Deprecated)**
   - Enable **Allow Remote Automation** in **Safari > Preferences > Developer** section if controlling Safari via WebDriver
3. **Display Resolution for 4K MacBooks or Monitors**
   - Adjust the display resolution to 1920x1080 (or the most similar ratio):
     - Navigate to **System Settings > Displays > Advanced...**
     - Select **Show resolutions as list**
     - Enable **Show all resolutions** and choose **1920x1200**
4. **Setup Wandb**
   - Click [here](https://wandb.ai/authorize), log in and copy your WandB API Key
   - Put it in a file named `secret_key.txt`, format it so it is seperated by a comma and no spaces like this:
   ```
   <organization name>,<wandb api key>
   ```
5. **Close All Tabs or add Offset**
   - Close all tabs and the safari browser fully
   or
   - Change the `tabs_present` in `game_config.json` to `1` instead of `0`

### Step 6: Run train.py
```
python train.py
```

### Notes
- turn on and off wandb by running the following commands
```
wandb enabled
```
```
wandb disabled
```
- You might need to adjust the threshold values for the template matching if the threshold value is too low
  - Press `cmd+shift+f` and search `similarity_threshold` (`1` of these) and `threshold` (`2` of these) there should be of of each in `fpa_env.py` and only one threshold in `track_swirlies.py`

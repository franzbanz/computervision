# Graphical Chart Reader using Computer Vision

This repository contains the code for my Bachelor's thesis project. The project involves developing a program that uses computer vision techniques and the OpenCV library for Python to read and interpret various types of graphical charts, such as those used in Simulink.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Graphical charts are widely used in engineering and scientific applications for modeling and simulation purposes. This project aims to automate the process of reading and interpreting these charts using computer vision. The primary focus is on Simulink-like charts, but the techniques developed can be applied to other types of graphical representations as well.

## Features

- Detection and extraction of different graphical elements from charts.
- Interpretation of connections and relationships between elements.
- Conversion of graphical data into a structured format for further analysis.
- Support for multiple types of charts and diagrams.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/graphical-chart-reader.git
cd graphical-chart-reader
pip install -r requirements.txt
```

## Usage

To use the program, run the main script with the path to the image of the chart you want to analyze:

```bash
python main.py --image path/to/your/chart.png
```

### Command Line Arguments

- `--image`: Path to the image file containing the chart.

## Technologies Used

- Python
- OpenCV
- Numpy
- Matplotlib (for visualization)

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

For any questions or further information, please contact me at franz@koehler-kn.de.
```
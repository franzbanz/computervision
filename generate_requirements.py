from pylatex import Command, Document, Section, Subsection
from pylatex.utils import NoEscape, italic
import os
import pathlib
import re

directory = r"/home/franz/Documents/computervision/testcases"

def generatePreamble():
    doc = Document(documentclass='article', document_options=['12pt'])

    # Add required packages
    packages = [
        'xcolor', 'hyperref', 'titletoc', 'float', 'caption',
        'subcaption', 'array', 'booktabs', 'graphicx', 'enumitem'
    ]
    for pkg in packages:
        doc.packages.append(NoEscape(r'\usepackage{' + pkg + '}') if isinstance(pkg, str) else pkg)
    NoEscape(r'\usepackage[margin=1in]{geometry}')

    return doc

def addTitlePage(doc):
    doc.append(NoEscape(r'''
\begin{titlepage}
    \centering
    % \includegraphics[width=0.24\textwidth]{logo.jpg}
    \par\vspace{2cm}

    {\Large \textbf{Institute of Aircraft Systems} \par}
    \vspace{0.5cm}
    {\large \textbf{University of Stuttgart} \par}
    \vspace{3cm}

    {\large \textbf{VVT Capabilities Document} \par}
    {\large Capabilities and Limits of the XGEE Visualization Verification Pipeline \par}
    {\large  \par}
    \vspace{3cm}

    {\large \textbf{Team} \par}
    \vspace{0.5cm}
    \begin{tabular}{ll}
    Franz Köhler & st174932@stud.uni-stuttgart.de \\
    \end{tabular}
    \par\vspace{3cm}

    {\large \today \par}
\end{titlepage}

\tableofcontents
\pagestyle{plain}
\newpage
'''))


def generateChapter(doc, sectionTitle, beforeImg, afterImg, warning_lines=None, label=None):
    with doc.create(Section(sectionTitle)):
        # Insert images side by side
        doc.append(NoEscape(r'\begin{figure}[H]'))
        doc.append(NoEscape(r'\centering'))

        doc.append(NoEscape(r'\begin{subfigure}[t]{0.49\textwidth}'))
        doc.append(NoEscape(r'\centering'))
        doc.append(NoEscape(fr'\includegraphics[width=\textwidth]{{{beforeImg}}}'))
        doc.append(NoEscape(r'\caption*{\textit{Input screenshot}}'))
        doc.append(NoEscape(r'\end{subfigure}'))

        doc.append(NoEscape(r'\hfill'))

        doc.append(NoEscape(r'\begin{subfigure}[t]{0.49\textwidth}'))
        doc.append(NoEscape(r'\centering'))
        doc.append(NoEscape(fr'\includegraphics[width=\textwidth]{{{afterImg}}}'))
        doc.append(NoEscape(r'\caption*{\textit{Output image with error indications}}'))
        doc.append(NoEscape(r'\end{subfigure}'))

        if label:
            doc.append(NoEscape(fr'\label{{{label}}}'))

        doc.append(NoEscape(r'\end{figure}'))

        if warning_lines:
            doc.append(NoEscape(r'\vspace{0.5cm}'))
            doc.append(NoEscape(r'\noindent\textbf{Warnings from vvt.log:} \\'))
            doc.append(NoEscape(r'\begin{itemize}[leftmargin=*, itemsep=0pt, topsep=0pt]'))
            for line in warning_lines:
                if "Starting of Model Comparison" in line:
                    continue  # Skip this line
                match = re.search(r'WARNING\s*[:\-]?\s*(.*)', line)
                if match:
                    warning_text = match.group(1).strip()
                    warning_text = warning_text.replace('_', r'\_').replace('&', r'\&')
                    doc.append(NoEscape(fr'\item \parbox[t]{{\dimexpr\linewidth-1em}}{{\ttfamily {warning_text}}}'))
            doc.append(NoEscape(r'\end{itemize}'))


        doc.append(NoEscape(r'\newpage'))


doc = generatePreamble()
addTitlePage(doc)

# Now iterate over sorted list
for root, dirs, files in os.walk(directory):
    dirs = sorted(dirs)
    image1, image2 = None, None
    warning_lines = []
    testcase = pathlib.PurePath(root)

    for file in files:
        if "input_image_after_preprocessing" in file:
            image1 = file
        elif "element_bbox_errors_labeled_colored" in file:
            image2 = file
        elif file == "vvt.log":
            with open(os.path.join(root, file), "r") as f:
                warning_lines = [line for line in f if "WARNING" in line]

    if image1 and image2:
        generateChapter(doc, testcase.name.replace("_", " ").title(), os.path.join(root, image1), os.path.join(root, image2), warning_lines=warning_lines )
doc.generate_pdf(clean_tex=False)
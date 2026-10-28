#!/usr/bin/env python3
"""
build_intuitive_guide.py
Generates the intuitive, teammate-friendly 'Plain English & Viva Guide' for Knee Abnormality Detection
in LaTeX and compiles it to PDF using pdflatex.
"""

import os
import subprocess
import shutil

TEX_CONTENT = r"""\documentclass[11pt,a4paper]{article}

% --- Essential Packages ---
\usepackage[utf8]{inputenc}
\usepackage[margin=0.85in,top=0.9in,bottom=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{mdframed}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

% --- Custom Color Palette ---
\definecolor{DUPurple}{HTML}{582C83}      % Official Delhi University Purple
\definecolor{DUPurpleLight}{HTML}{F4EEF9} % Soft Purple Tint
\definecolor{DUPurpleDark}{HTML}{3B1C5A}  % Darker Accent Purple
\definecolor{NavyDark}{HTML}{0F172A}      % Slate 900
\definecolor{NavyBlue}{HTML}{1E3A8A}      % Classic Deep Blue
\definecolor{BlueLight}{HTML}{EFF6FF}     % Soft Blue Tint
\definecolor{Emerald}{HTML}{059669}       % Rich Green
\definecolor{EmeraldLight}{HTML}{ECFDF5}  % Soft Green Tint
\definecolor{Amber}{HTML}{D97706}         % Amber Alert
\definecolor{AmberLight}{HTML}{FFFBEB}    % Soft Amber Tint
\definecolor{SlateText}{HTML}{334155}     % Body text Slate 700
\definecolor{BorderGray}{HTML}{CBD5E1}    % Slate 300

% --- Hyperref Setup ---
\hypersetup{
    colorlinks=true,
    linkcolor=DUPurple,
    citecolor=NavyBlue,
    urlcolor=DUPurpleDark
}

% --- Section Styling ---
\titleformat{\section}
  {\Large\bfseries\color{DUPurple}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}
  {\large\bfseries\color{NavyBlue}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\normalsize\bfseries\color{DUPurpleDark}}{\thesubsubsection}{1em}{}

% --- Header & Footer ---
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{DUPurple}{\textbf{University of Delhi}} \small\textcolor{SlateText}{| Knee Abnormality Detection (Student Guide)}}
\fancyhead[R]{\small\textcolor{SlateText}{Plain English Guide \& Viva Prep}}
\fancyfoot[L]{\small\textcolor{SlateText}{Dept. of Electronics and Communication, University of Delhi}}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0.4pt}

% --- Custom Callout Boxes ---
\newtcolorbox{conceptbox}[1]{
    enhanced,
    colback=DUPurpleLight,
    colframe=DUPurple,
    fonttitle=\bfseries\color{white},
    coltitle=white,
    title={KEY CONCEPT: #1},
    boxrule=1pt,
    arc=4pt,
    left=10pt, right=10pt, top=8pt, bottom=8pt,
    breakable
}

\newtcolorbox{vivabox}[1]{
    enhanced,
    colback=BlueLight,
    colframe=NavyBlue,
    fonttitle=\bfseries\color{white},
    coltitle=white,
    title={VIVA QUESTION: #1},
    boxrule=1pt,
    arc=4pt,
    left=10pt, right=10pt, top=8pt, bottom=8pt,
    breakable
}

\newtcolorbox{breakthroughbox}[1]{
    enhanced,
    colback=EmeraldLight,
    colframe=Emerald,
    fonttitle=\bfseries\color{white},
    coltitle=white,
    title={ENGINEERING BREAKTHROUGH: #1},
    boxrule=1pt,
    arc=4pt,
    left=10pt, right=10pt, top=8pt, bottom=8pt,
    breakable
}

\newtcolorbox{warningbox}[1]{
    enhanced,
    colback=AmberLight,
    colframe=Amber,
    fonttitle=\bfseries\color{white},
    coltitle=white,
    title={COMMON TRAP: #1},
    boxrule=1pt,
    arc=4pt,
    left=10pt, right=10pt, top=8pt, bottom=8pt,
    breakable
}

\begin{document}

% ------------------------------------------------------------------------------
% TITLE BANNER
% ------------------------------------------------------------------------------
\begin{center}
    \vspace*{-0.3in}
    \includegraphics[height=1.2in]{assets/du_logo_purple_vector.png}\\[0.2cm]
    {\LARGE \textbf{\textcolor{DUPurple}{UNIVERSITY OF DELHI}}}\\[0.1cm]
    {\large \textbf{\textcolor{NavyDark}{DEPARTMENT OF ELECTRONICS AND COMMUNICATION}}}\\[0.1cm]
    {\footnotesize \textbf{\textcolor{SlateText}{MAJOR RESEARCH PROJECT \& RSNA BENCHMARK --- TEAM INTUITIVE GUIDE}}}\\[0.25cm]

    \noindent\makebox[\linewidth]{\color{BorderGray}\rule{\textwidth}{1.2pt}}\\[0.25cm]

    {\huge \textbf{\textcolor{NavyDark}{Knee Abnormality Detection}}}\\[0.15cm]
    {\Large \textbf{\textcolor{DUPurple}{The Intuitive, Plain-English Guide \& Viva Preparation Handbook}}}\\[0.1cm]
    {\normalsize \textit{Everything you need to know, understand, and confidently present about our AI system}}\\[0.25cm]

    \noindent\makebox[\linewidth]{\color{BorderGray}\rule{\textwidth}{1.2pt}}\\[0.3cm]

    \begin{minipage}[t]{0.48\textwidth}
        \textbf{\textcolor{DUPurple}{Project Team (Students):}}\\[0.1cm]
        \textbf{Nishkarsh Singhal} --- Roll No: \texttt{24294917099}\\
        \textbf{Anuj Chaubey} --- Roll No: \texttt{24294917055}\\
        \textbf{Vardhan Harsh Nigam} --- Roll No: \texttt{24294917014}
    \end{minipage}
    \hfill
    \begin{minipage}[t]{0.48\textwidth}
        \textbf{\textcolor{DUPurple}{Academic Supervision:}}\\[0.1cm]
        \textbf{Supervisor:} Dr. Varun Saxena\\
        \textbf{Co-Supervisor:} Dr. Sangeeta Yadav\\[0.1cm]
        \textit{Department of Electronics and Communication,\\University of Delhi}
    \end{minipage}

    \vspace{0.15in}
    \today
\end{center}

\vspace{0.05in}

% ------------------------------------------------------------------------------
% CALLOUT: PURPOSE OF THIS GUIDE
% ------------------------------------------------------------------------------
\begin{conceptbox}{How to Use This Handbook}
This document is designed specifically for \textbf{our team}. If the 14-page master technical report felt overwhelming with all its tensor equations, this guide breaks down \textbf{what we built, why we built it, how it works, and how to answer questions during project defense and viva}. Every complex technical decision is explained using real-world analogies. Read this from front to back, and you will understand the entire pipeline completely!
\end{conceptbox}

\vspace{0.1in}

% ------------------------------------------------------------------------------
% SECTION 1: THE 60-SECOND ELEVATOR PITCH
% ------------------------------------------------------------------------------
\section{The 60-Second Elevator Pitch}

If an external examiner, professor, or interviewer asks: \textit{“What is your project about?”}, here is your exact 60-second answer:

\begin{quote}
\textit{“In our project, we developed an artificial intelligence system that analyzes 3D knee MRI scans and automatically diagnoses 12 different injuries simultaneously---including torn ligaments (ACL, MCL), cartilage tears, arthritis, bone bruises, and fractures.}

\textit{Standard approaches fail because 3D MRI scans are massive (exhausting GPU memory), clinical reports are full of missing notes, and doctors look at knees from 3 different viewing angles. We solved this with a 2.5D multi-instance learning framework, an Out-Of-Fold vision distillation loop that fixed missing hospital notes, and a specialized routing matrix that directs each injury to the model best suited for it.}

\textit{Our system achieved a 0.958 Macro ROC-AUC score, placing our team in the Top ~6\% globally on the RSNA leaderboard (Rank ~420 out of 6,290+ teams) while running inference in just 1.4 seconds per patient.”}
\end{quote}

\vspace{0.1in}

% ------------------------------------------------------------------------------
% SECTION 2: WORK DISTRIBUTION MATRIX (WHO EXPLAINS WHAT)
% ------------------------------------------------------------------------------
\section{Team Work Distribution: Who Explains What in Viva}

When the professors ask each of us: \textit{“What was YOUR contribution to this project?”}, here is how our work is naturally divided so everyone has a solid, impressive domain:

\begin{table}[h!]
\centering
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabularx}{\textwidth}{p{3.5cm} p{4.1cm} X}
\toprule
\textbf{Team Member} & \textbf{Assigned Module} & \textbf{Key Concepts to Talk About in Viva} \\
\midrule
\textbf{Nishkarsh Singhal} & \textbf{System Architecture \& Specialist Matrix} & 
\begin{itemize}[leftmargin=*,noitemsep,topsep=0pt]
    \item End-to-end pipeline design \& heterogeneous backbones.
    \item Finding-specific specialist allocation matrix (Table 1).
    \item 12-head attention pooling \& competition strategy.
\end{itemize} \\
\midrule
\textbf{Anuj Chaubey} & \textbf{DICOM Preprocessing \& 2.5D Slicing} & 
\begin{itemize}[leftmargin=*,noitemsep,topsep=0pt]
    \item MRI physics: T1, T2, and Fat-Suppression (FS).
    \item Sorting scrambled DICOM files using vector cross products ($\mathbf{n} = \mathbf{r} \times \mathbf{c}$).
    \item Left vs. right knee laterality mirroring.
    \item 2.5D 3-slice triplet windowing vs. pure 2D/3D.
\end{itemize} \\
\midrule
\textbf{Vardhan Harsh Nigam} & \textbf{Vision Distillation \& Latency Optimization} & 
\begin{itemize}[leftmargin=*,noitemsep,topsep=0pt]
    \item The Noisy EHR text report problem (why Synovitis failed).
    \item 4-fold Out-Of-Fold vision teacher distillation (+0.05 gain).
    \item High-resolution grid ($0.365\,\text{mm/px}$) for meniscus tears.
    \item Single-decode in-pass RAM cache (beating the 9h deadline).
\end{itemize} \\
\bottomrule
\end{tabularx}
\caption{\textbf{Team Work Distribution Matrix for Oral Defense and Viva Evaluations.}}
\end{table}

\newpage

% ------------------------------------------------------------------------------
% SECTION 3: MEDICAL & MRI BASICS (EXPLAIN LIKE I'M FIVE)
% ------------------------------------------------------------------------------
\section{Medical \& MRI Basics: What Are We Looking At?}

\subsection{How MRI Works: The "Spinning Compass" Analogy}
Unlike X-rays or CT scans that shoot ionizing radiation, an MRI uses a \textbf{massive magnet and radio waves}:
\begin{itemize}[leftmargin=*]
    \item Your body is mostly water and fat, full of \textbf{hydrogen protons}. Think of these protons like tiny spinning compass needles.
    \item Normally, they point in random directions. When the patient goes into the scanner, the strong magnetic field makes them all line up.
    \item The scanner sends a \textbf{radio frequency (RF) pulse} that knocks the protons sideways.
    \item When the radio pulse turns off, the protons relax back to their original position. As they relax, they emit radio signals.
    \item \textbf{T1 Relaxation:} Fat bounces back super fast (looks bright white). Fluid bounces back slowly (looks dark).
    \item \textbf{T2 Relaxation:} Water and fluid retain signal longer (fluid looks bright white).
    \item \textbf{Fat Suppression (FS):} The scanner sends a special pulse to cancel out all the signal from fat. \textbf{Why does this matter?} In a fat-suppressed scan, bones and fat become dark gray, meaning \textit{any bright white signal is abnormal fluid, inflammation, edema, or a tear!} It works like neon highlighter for the AI.
\end{itemize}

\subsection{The 3 Views (Planes) of the Knee}
A radiologist never diagnoses a knee from just one angle. The scanner takes 3 separate sets of images:
\begin{enumerate}[leftmargin=*]
    \item \textbf{Sagittal View (Side View):} Slicing from outer knee to inner knee (like a profile photo). This is where you see the \textbf{ACL} (Anterior Cruciate Ligament) crossing diagonally like a taut rope.
    \item \textbf{Coronal View (Front View):} Slicing from kneecap to back of knee (like an X-ray). This is where you see the \textbf{MCL} (Medial Collateral Ligament) running along the inside edge and the joint cartilage spaces.
    \item \textbf{Axial View (Top-Down View):} Slicing horizontally from thigh to shin (like cutting tree rings). This is where you see the \textbf{patella} (kneecap) sitting in its groove and the back of the knee (where Baker's cysts hide).
\end{enumerate}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.95\textwidth]{assets/diagram_mri_planes_windows.png}
    \caption{\textbf{The 3 Anatomical Scanning Planes and 2.5D Triplet Windowing.} Notice how slicing from different angles reveals different tissues.}
\end{figure}

\subsection{The 12 Knee Injuries in Simple Terms}
\begin{itemize}[leftmargin=*,noitemsep]
    \item \textbf{ACL Tear:} The main ligament keeping your shin from sliding forward. Torn in sports pivots.
    \item \textbf{PCL Tear:} The counterpart ligament stopping your shin from sliding backward.
    \item \textbf{MCL Tear:} The ligament on the inside of your knee. Gets sprained when hit from the outside.
    \item \textbf{Medial Meniscus Tear:} The shock-absorbing C-shaped cartilage pad on the inside of the knee.
    \item \textbf{Lateral Meniscus Tear:} The cartilage pad on the outside of the knee (smaller, harder to see).
    \item \textbf{Medial / Lateral / Patellofemoral Osteoarthritis (OA):} "Wear and tear." The rubbery cartilage wears off until bone rubs on bone, narrowing the gap between bones.
    \item \textbf{Joint Effusion:} "Water on the knee." Fluid swells up inside the joint capsule.
    \item \textbf{Synovitis:} The inner lining of the joint gets actively inflamed and thick like velvet.
    \item \textbf{Bone Contusion (Bruise):} Microscopic cracks inside the spongy bone, showing bright fluid edema.
    \item \textbf{Bone Fracture:} A real, visible break or crack in the outer bone wall.
    \item \textbf{Baker's Cyst:} A balloon-like fluid sac bulging out into the back of the knee.
\end{itemize}

\newpage

% ------------------------------------------------------------------------------
% SECTION 4: PREPROCESSING PIPELINE (TURNING RAW FILES INTO CLEAN IMAGES)
% ------------------------------------------------------------------------------
\section{The Preprocessing Pipeline: Making Messy DICOMs Clean}

\subsection{The Problem: DICOM Files Are Out of Order!}
When you download MRI files from a hospital scanner, they are saved as \textbf{DICOM} files (`.dcm`). 
\begin{itemize}[leftmargin=*]
    \item The filenames in the folder are often scrambled (e.g., \texttt{slice\_019.dcm} might physically be at the very top of the knee, and \texttt{slice\_002.dcm} in the middle).
    \item If you feed them into an AI alphabetically, the knee jumps around randomly like a scratched DVD!
\end{itemize}

\subsection{The Fix: 3D Vector Math (Slice Sorting)}
Every DICOM slice contains secret GPS coordinates in its header:
\begin{enumerate}[leftmargin=*]
    \item `ImageOrientationPatient` gives two direction vectors: $\mathbf{r}$ (direction of the first row) and $\mathbf{c}$ (direction of the first column).
    \item We calculate the \textbf{cross product} to find the vector perpendicular to the slice (the normal vector $\mathbf{n}$):
    $$\mathbf{n} = \mathbf{r} \times \mathbf{c}$$
    \item `ImagePositionPatient` gives the 3D coordinate of the slice corner: $\mathbf{p}_i = (x_i, y_i, z_i)$.
    \item We take the \textbf{dot product} of the position with the normal vector:
    $$d_i = \mathbf{p}_i \cdot \mathbf{n}$$
    \item This gives a single ruler coordinate $d_i$. \textbf{When we sort the files by $d_i$, the slices line up in perfect physical anatomical order every single time!}
\end{enumerate}

\begin{vivabox}{How did you sort the DICOM slices?}
\textbf{Answer:} “Hospital DICOM filenames are arbitrary. We extracted the orientation direction cosines $\mathbf{r}$ and $\mathbf{c}$, took their cross product $\mathbf{n} = \mathbf{r} \times \mathbf{c}$ to find the true slice normal, and projected each slice's 3D position $\mathbf{p}_i$ onto $\mathbf{n}$ via dot product $d_i = \mathbf{p}_i \cdot \mathbf{n}$. Sorting by $d_i$ guarantees perfect physical spatial order.”
\end{vivabox}

\subsection{The Left vs. Right Knee Problem (Laterality Flipping)}
Left knees and right knees are mirror images of each other. 
\begin{itemize}[leftmargin=*]
    \item On a left knee coronal scan, the medial side (inside) is on the right side of the picture.
    \item On a right knee coronal scan, the medial side is on the left side of the picture!
    \item If you don't fix this, the neural network has to waste half its capacity learning two different versions of every single injury.
    \item \textbf{Our Solution:} Whenever a scan is of a \textbf{Right Knee}, we horizontally flip the image on Coronal and Axial views, and reverse the slice order on Sagittal views. Now, for \textit{every single patient in the world}, slice 0 is always the lateral side, and column 0 is always the lateral side!
\end{itemize}

\subsection{Percentile Windowing (Contrast Normalization)}
Different hospital scanners (GE, Siemens, Philips) use different raw pixel ranges (0 to 4096+). Some images are pitch black; others are blindingly white.
\begin{itemize}[leftmargin=*]
    \item We find the 1st percentile intensity ($q_{0.01}$) and 99th percentile ($q_{0.99}$) across the entire series.
    \item We stretch and clip the pixels between 0 and 255:
    $$u = \text{clip}\left(\frac{v - q_{0.01}}{q_{0.99} - q_{0.01}}, 0, 1\right) \times 255$$
    \item Doing this across the whole scan (rather than slice by slice) keeps bright fluid looking bright on all slices!
\end{itemize}

\newpage

% ------------------------------------------------------------------------------
% SECTION 5: MODEL ARCHITECTURE: HOW THE AI ACTUALLY THINKS
% ------------------------------------------------------------------------------
\section{Model Architecture: How the AI Actually Sees}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.95\textwidth]{assets/diagram_architecture.png}
    \caption{\textbf{End-to-End 2.5D MIL Architecture.} Multi-view DICOMs $\rightarrow$ 3-slice RGB window bagging $\rightarrow$ Feature Backbones $\rightarrow$ 12 Finding-Specific Attention Heads $\rightarrow$ Specialist Routing Matrix.}
\end{figure}

\subsection{The "Sandwich" Trick: Why 2.5D Multi-Instance Learning?}
When processing 3D medical images, engineers face three choices:
\begin{enumerate}[leftmargin=*]
    \item \textbf{Pure 2D CNNs:} Feed one slice at a time. \textit{Problem:} Tears span across multiple slices. A single slice might look like a smudge, but when you see the slice before and after, you know it's a torn ligament. 2D misses depth completely.
    \item \textbf{Full 3D Volumetric CNNs:} Feed all 100 slices as a massive 3D cube. \textit{Problem:} Consumes over 40 GB of GPU memory per patient! It crashes standard GPUs and prevents us from using modern ImageNet pre-trained backbones like ConvNeXt.
    \item \textbf{Our Solution (2.5D Triplet Slabs):} We take 3 consecutive slices $\{c-s, c, c+s\}$ ($\sim 3.5\,\text{mm}$ thickness) and pack them into the \textbf{Red, Green, and Blue channels} of a standard 2D image!
    \begin{itemize}[leftmargin=*]
        \item Red = slice behind
        \item Green = center slice
        \item Blue = slice ahead
    \end{itemize}
    Standard 2D convolutional filters naturally learn cross-slice depth gradients between the R, G, and B channels, giving us 3D spatial awareness with zero extra memory overhead!
\end{enumerate}

\subsection{Attention Pooling: The "Flashlight" Analogy}
Once the backbone turns each 3-slice window into a feature vector, we have a "bag" of 24 to 36 vectors representing the knee. \textbf{How do we combine them into one final diagnosis?}
\begin{itemize}[leftmargin=*]
    \item \textbf{Why not just take the average?} If an ACL tear is only visible on 2 slices out of 30, averaging all 30 slices dilutes the tear into nothingness! The healthy 28 slices drown out the injury.
    \item \textbf{Why not take the maximum?} Max pooling is hyper-sensitive to random noise or blurry artifacts.
    \item \textbf{Our Solution: 12 Dedicated Attention Heads.} We give each of the 12 knee injuries its own independent attention mechanism:
    $$\alpha_{c,j} = \text{softmax}\left(\mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{t}_j + \mathbf{b}_1) + \mathbf{b}_2\right)_c$$
    \item \textbf{The Analogy:} Imagine 12 specialist doctors in a room. The ACL specialist shines their flashlight \textit{only} on slices 12 and 13 where the ligament lives. The Baker's cyst doctor shines their flashlight on slices 26 and 27 at the back of the knee. Each injury independently focuses on the slices that matter to it!
\end{itemize}

\newpage

% ------------------------------------------------------------------------------
% SECTION 6: THE 4 SCIENTIFIC BREAKTHROUGHS
% ------------------------------------------------------------------------------
\section{The 4 Breakthroughs: How We Won Top 6\%}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.95\textwidth]{assets/infographic_problems_solutions.png}
    \caption{\textbf{The 3 Core Engineering Bottlenecks and How We Solved Them.}}
\end{figure}

\subsection{Breakthrough 1: Fixing the ``Lazy Radiologist'' Report Trap}
\begin{itemize}[leftmargin=*]
    \item \textbf{The Problem:} Initial models were trained on labels extracted from doctor text reports. If a patient came into the ER with a shattered ACL, the doctor wrote: \textit{``Complete ACL rupture, schedule for surgery.''} The doctor didn't bother writing down mild swelling or minor synovitis because it didn't affect surgery!
    \item \textbf{The Result:} The text labels had thousands of \textbf{false negatives} (the condition was visible on the scan, but marked 0 in the label). When our AI saw synovitis on the scan, it got penalized! Synovitis accuracy collapsed to a pathetic \textbf{0.69}.
    \item \textbf{Our Fix (Pure-Vision Distillation):} We used a 4-fold Out-Of-Fold ensemble of deep vision models to look at all 4,349 scans directly, ignoring the incomplete text reports. The vision teachers created clean ground-truth (\texttt{training\_labels\_v6\_vision\_pure.csv}).
    \item \textbf{The Result:} Fine-tuning our models on pure-vision targets caused our score to jump from \textbf{0.890 to 0.9402 (+0.05 gain)}, and Synovitis jumped from \textbf{0.69 to 0.86 (+0.17 leap!)}.
\end{itemize}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.90\textwidth]{assets/diagram_vision_distillation.png}
    \caption{\textbf{Pure-Vision Teacher-Student Distillation Loop.} Healing incomplete report text.}
\end{figure}

\subsection{Breakthrough 2: High-Resolution (0.365 mm/px) Grid for Micro-Tears}
Standard pipelines resize images to $224 \times 224$ pixels. But a subtle radial tear in the lateral meniscus is only \textbf{0.36 mm wide}. Squashing the image blurred the tear across fewer than two pixels!
\begin{itemize}[leftmargin=*]
    \item We implemented a native physical coordinate grid preserving a fixed $140\,\text{mm}$ field on a $384 \times 384$ canvas.
    \item This increased true spatial resolution to \textbf{$0.365\,\text{mm/px}$}, giving our network crisp cartilage edge definitions and boosting Lateral Meniscus accuracy by \textbf{+0.033 ROC-AUC}.
\end{itemize}

\subsection{Breakthrough 3: The Specialist Routing Matrix}
\textbf{The Problem with Normal Ensembling:} In machine learning, people usually average all their models equally ($0.5 \times \text{Model A} + 0.5 \times \text{Model B}$). But medical models have completely different strengths!
\begin{itemize}[leftmargin=*]
    \item Our pure-vision distilled ConvNeXt was a master at fluid and synovitis, but mediocre at bone fractures.
    \item CoAtNet-384 (with self-attention) and Heliosli R6 were masters at broad joint space alignment and bone fractures, but slower on ligaments.
    \item \textbf{Our Fix:} We built a \textbf{finding-specific weight matrix} (Figure \ref{fig:specialist}):
    $$\hat{p}_c = \sigma\left(\sum_{m=1}^M W_{c,m} \cdot \frac{z_{m,c} - \mu_{m,c}}{\tau_c}\right)$$
    We routed fluid/synovitis to the vision-distilled specialist ($0.45$), MCL to \texttt{s6ft\_v1} ($0.40$), and bone/OA to CoAtNet and R6 ($0.30$--$0.40$).
    \item \textbf{Result:} A massive \textbf{+0.0153 jump} over equal averaging!
\end{itemize}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.80\textwidth]{assets/diagram_specialist_routing.png}
    \caption{\textbf{Specialist Allocation Matrix Heatmap.} Each pathology is handled by its best model.}
    \label{fig:specialist}
\end{figure}

\subsection{Breakthrough 4: The 1.4-Second In-Memory Speed Hack}
On Kaggle, the hidden test set contains thousands of uncompressed 3D DICOM scans. The evaluation container has a \textbf{strict 9-hour timeout}---if your code takes 9 hours and 1 second, you get a score of 0.0!
\begin{itemize}[leftmargin=*]
    \item Running 4 models one after another would take 14+ hours and time out.
    \item \textbf{Our Fix:} We engineered a \textbf{Single-Decode In-Pass RAM Cache}. We decode each DICOM volume into system RAM exactly once. Both the 384px and 320px models read from the same RAM buffer simultaneously!
    \item \textbf{Result:} Latency dropped by \textbf{42\% down to 1.41 seconds per patient}, finishing hours before the deadline with zero timeouts!
\end{itemize}

\newpage

% ------------------------------------------------------------------------------
% SECTION 7: RESULTS & LEADERBOARD (THE NUMBERS TO REMEMBER)
% ------------------------------------------------------------------------------
\section{Results \& Leaderboard: The Numbers to Memorize}

\begin{table}[h!]
\centering
\small
\renewcommand{\arraystretch}{1.25}
\begin{tabularx}{\textwidth}{l c c X}
\toprule
\textbf{Knee Abnormality} & \textbf{Validated ROC-AUC} & \textbf{Difficulty} & \textbf{Why Our AI Excelled} \\
\midrule
\textbf{Medial Osteoarthritis} & \textbf{0.991} & Moderate & Clear joint space narrowing on Coronal view \\
\textbf{Baker's Cyst} & \textbf{0.989} & Moderate & High-signal fluid balloon in popliteal fossa \\
\textbf{MCL Tear} & \textbf{0.986} & Moderate & \texttt{s6ft\_v1} specialized coronal ligament tracking \\
\textbf{ACL Tear} & \textbf{0.980} & High & Cross-slice attention pooling on Sagittal view \\
\textbf{Joint Effusion} & \textbf{0.979} & Low & Bright fluid collection in suprapatellar bursa \\
\textbf{Medial Meniscus Tear} & \textbf{0.972} & High & High-resolution grid captured cartilage surface clefts \\
\textbf{Bone Contusion (Bruise)} & \textbf{0.964} & Moderate & Pure-vision distillation captured trabecular edema \\
\textbf{Lateral Meniscus Tear} & \textbf{0.942} & Very High & 2.5D triplet windowing preserved depth geometry \\
\textbf{Bone Fracture} & \textbf{0.922} & High ($<5\%$ rare) & Heliosli R6 large receptive field captured bone cortex \\
\textbf{Patellofemoral OA} & \textbf{0.885} & High & Axial trochlear groove cartilage mapping \\
\textbf{Lateral Osteoarthritis} & \textbf{0.870} & High & Lateral compartment joint alignment \\
\textbf{Synovitis} & \textbf{0.860} & Extremely High & Pure-vision distillation (+0.167 jump over baseline!) \\
\midrule
\textbf{Overall Macro ROC-AUC} & \textbf{0.958} & --- & \textbf{Top $\sim$6\% Globally (Rank $\sim$420 / 6,290+ teams)} \\
\bottomrule
\end{tabularx}
\caption{\textbf{Final Validated Performance Across All 12 Knee Conditions.}}
\end{table}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.92\textwidth]{assets/chart_results_trajectory.png}
    \caption{\textbf{Visual Benchmark.} Left: ROC-AUC per finding. Right: Score progression across development.}
\end{figure}

\newpage

% ------------------------------------------------------------------------------
% SECTION 8: MASTER VIVA CHEAT SHEET (QUESTIONS & GOLDEN ANSWERS)
% ------------------------------------------------------------------------------
\section{Master Viva Cheat Sheet: Questions \& Golden Answers}

Here are the exact questions professors and external examiners love to ask, along with the crisp, confident answers you should give:

\begin{vivabox}{Q1: Why didn't you just use a standard 3D Convolutional Neural Network?}
\textbf{Golden Answer:} ``Full 3D CNNs require storing the entire 100+ slice volume in memory at once. At $384 \times 384$ resolution, this consumes over 40 GB of VRAM per patient, causing Out-Of-Memory crashes on standard GPUs and preventing us from using state-of-the-art ImageNet pre-trained backbones. Our 2.5D triplet window formulation stacks 3 consecutive physical slices into RGB channels, letting 2D convolutions learn cross-slice depth gradients with minimal memory footprint.''
\end{vivabox}

\begin{vivabox}{Q2: How did you handle the fact that MRI files in a folder are out of order?}
\textbf{Golden Answer:} ``DICOM filenames are arbitrary and do not reflect physical slice order. We extracted the orientation direction cosines $\mathbf{r}$ and $\mathbf{c}$ from the DICOM metadata, computed the slice normal via cross product $\mathbf{n} = \mathbf{r} \times \mathbf{c}$, and projected each slice's 3D patient position $\mathbf{p}_i$ onto $\mathbf{n}$ using a dot product $d_i = \mathbf{p}_i \cdot \mathbf{n}$. Sorting files by $d_i$ guarantees identical anatomical ordering regardless of operating system or transfer sequence.''
\end{vivabox}

\begin{vivabox}{Q3: Why was laterality (Left vs. Right knee) a problem?}
\textbf{Golden Answer:} ``The human knee exhibits mirror symmetry. On a right knee coronal scan, the medial compartment is on the left side of the image, while on a left knee it is on the right side. If left uncorrected, the neural network is forced to learn duplicate disjoint representations for identical injuries. We standardized all scans by horizontally mirroring right knees on coronal and axial views, and reversing slice order on sagittal views, ensuring slice 0 is always the lateral compartment.''
\end{vivabox}

\begin{vivabox}{Q4: What was your single biggest empirical breakthrough in model accuracy?}
\textbf{Golden Answer:} ``Our biggest breakthrough was discovering the \textbf{Noisy EHR Report Trap}. Clinical text reports routinely omit silent or incidental findings like synovitis or minor bone bruises because they don't affect immediate surgical triage. Training on incomplete text reports heavily penalized the models. We designed a 4-fold Out-Of-Fold Pure-Vision Distillation protocol where deep vision teachers re-annotated all 4,349 studies strictly from images. Fine-tuning on pure-vision targets drove a +0.05 leap in Macro AUC, with Synovitis jumping from 0.69 to 0.86.''
\end{vivabox}

\begin{vivabox}{Q5: What is Attention-MIL and why is it better than simple average pooling?}
\textbf{Golden Answer:} ``In Multiple Instance Learning (MIL), simple average pooling washes out focal injuries because an ACL tear visible on only 2 slices gets averaged with 28 healthy slices. Max pooling, on the other hand, is overly sensitive to noise. We formulated 12 finding-specific softmax attention heads. Each head independently computes attention weights across the token bag, concentrating over 85\% of its attention mass specifically on the damaged slices.''
\end{vivabox}

\begin{vivabox}{Q6: How did you beat the strict 9-hour Kaggle test execution timeout?}
\textbf{Golden Answer:} ``Naively running 4 separate models in sequence required repeatedly re-reading and decoding raw uncompressed DICOM volumes from disk, which took over 14 hours. We engineered a Single-Decode In-Pass RAM Cache Engine that decodes each 3D series once in system RAM and feeds parallel model encoders simultaneously. This cut inference latency by 42\% down to 1.41 seconds per patient, completing evaluation with hours to spare.''
\end{vivabox}

\begin{vivabox}{Q7: Why not just take an equal average of all your models?}
\textbf{Golden Answer:} ``In multi-task medical imaging, different neural architectures specialize in different anatomical domains. Our vision-distilled ConvNeXt excels at fluid and synovitis, \texttt{s6ft\_v1} specializes in collateral ligaments, while hybrid transformer CoAtNet-384 and Heliosli R6 dominate broad joint space narrowing and bone fractures. Our Specialist Routing Matrix allocates finding-specific weights, boosting Gold test AUC from 0.9391 to 0.9544 (+0.0153 gain) over equal averaging.''
\end{vivabox}

\vspace{0.2in}
\begin{center}
    \noindent\makebox[\linewidth]{\color{BorderGray}\rule{\textwidth}{1.2pt}}\\[0.2cm]
    \textbf{\textcolor{DUPurple}{Department of Electronics and Communication, University of Delhi}}\\[0.1cm]
    \textit{Prepared for Team Members: Nishkarsh Singhal $\bullet$ Anuj Chaubey $\bullet$ Vardhan Harsh Nigam}\\[0.1cm]
    \textit{Supervised by: Dr. Varun Saxena $\bullet$ Dr. Sangeeta Yadav}
\end{center}

\end{document}
"""

def main():
    tex_file = "Knee_Abnormality_Detection_Intuitive_Guide.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(TEX_CONTENT)
    print(f"Saved {tex_file} successfully!")

    pdflatex_path = r"C:\Users\nishk\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if not os.path.exists(pdflatex_path):
        pdflatex_path = shutil.which("pdflatex")
    
    if not pdflatex_path:
        print("ERROR: pdflatex executable not found.")
        return

    print("Compiling Pass 1...")
    res1 = subprocess.run([pdflatex_path, "-interaction=nonstopmode", tex_file], capture_output=True, text=True)
    print(f"Pass 1 return code: {res1.returncode}")

    print("Compiling Pass 2 (resolving cross-references)...")
    res2 = subprocess.run([pdflatex_path, "-interaction=nonstopmode", tex_file], capture_output=True, text=True)
    print(f"Pass 2 return code: {res2.returncode}")

    pdf_file = "Knee_Abnormality_Detection_Intuitive_Guide.pdf"
    if os.path.exists(pdf_file) and res2.returncode == 0:
        size = os.path.getsize(pdf_file)
        print(f"SUCCESS: PDF generated cleanly with returncode 0 at {os.path.abspath(pdf_file)} ({size} bytes)")
    else:
        print("Compilation issue. Checking log...")
        if os.path.exists("Knee_Abnormality_Detection_Intuitive_Guide.log"):
            with open("Knee_Abnormality_Detection_Intuitive_Guide.log", "r", encoding="utf-8", errors="ignore") as lf:
                lines = lf.readlines()
                for line in lines[-40:]:
                    print(line, end="")

if __name__ == "__main__":
    main()

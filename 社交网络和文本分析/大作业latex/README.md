# 敬业修改：写在之前
* 本模板的figure文件夹已经被挪到了tex文件夹下面，这样就可以正确读取
* 本模板的main.tex文件是用来生成主要内容的，增加\usepackage{ctex}即可正确输入中文
* 本模板的naturestyle.sty文件用来生成主要的标题、作者信息等。增加\usepackage{ctex}即可正确输入中文
* 已经将main.pdf复制一份放在了此目录下，以防老师迷路。

# nature-tex
LaTeX template and build code for a Nature style research article submission

# Features
* Keeps your tex organized into separate `main.tex` and `sup.tex` files
* Preamble kept separate in `naturetex.sty` for cleaner tex code
* Allows you to use a separate bibliography file with `cite` commands as usual
* Generates the final single `.tex` file for you, as expected by Nature journals
* **Works in shared editors like Overleaf**
    * You can continue to edit and compile your tex files in Overleaf. How: clone the backing repo for a blank Overleaf project, and push the files of this repo into it. Then you can edit your files in Overleaf normally in the cloud, but run this tool locally to get submission-ready files.

# Prerequisites
* Python
* A LaTeX distribution (You'll need `pdflatex`, `biber`, and `bibtex` in your path)

# Usage
1. Clone this repository
2. `$ cd nature-tex`
3. `$ make`
4. (Optional) You can use `$ make clean` to remove the generated temporary files.

This should have generated two folders, `submit` and `proof`. Inspect those to make sure everything is working.

```
proof/      # CONVENIENCE FILES, DO NOT SUBMIT TO NATURE
  main.pdf  # The the main manuscript as a pdf, with the figures included, for easy reading
  
submit/     # FILES READY FOR SUBMISSION TO NATURE
  main.tex  # The main manuscript tex, all as a single file, no figures
  sup.pdf   # The supplement as a pdf
```

Now, edit `tex/main.tex`, `tex/sup.tex`, and `tex/bibliography.bib` to your heart's desire, and replace the dummy tex with real science!
## TeX rules
To allow proper cross referencing between the main manuscript and supplement, you must use the labels specified in `cross_refs.json`

You may edit `cross_refs.json` to add/remove any keys you might want (e.g. Notes, Box, etc)

Each (key,value) pair in `cros_refs.json` specifies a label type which will be kept track of and a full string which will be printed. If you use any of those labels in either the main manuscript or supplement, the python code will correctly keep track of it and replace it with a correctly numbered string.

For example:
```
% main.tex
\label{fig:my_cool_fig}
Hey, check out this result in \ref{fig:my_cool_fig} and \ref{sup.fig:my_extra_fig}!
% sup.tex
\label{sup.fig:my_extra_fig}
This extra fig is kinda like \ref{fig:my_cool_fig}.
```
Will be processed into:
```
% main.tex
\label{fig:my_cool_fig}
Hey, check out this result in Figure 1 and Supplementary Figure 1!
% sup.tex
\label{sup.fig:my_extra_fig}
This extra fig is kinda like Figure 1.
```

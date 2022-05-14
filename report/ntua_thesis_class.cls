\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ntua_thesis_class}[2022/03/10 My custom M.Eng. thesis class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[]{report}

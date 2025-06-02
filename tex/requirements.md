# Requirements Document

## Different types of error indications:

- **Tokenization** Untokenized pixels
- **Syntactical analysis** Missing parent element
- **Model instantiation** Element is not instatiated - association cannot connect
- **Model comparison** Element is missing in recognized model

## Possible error types

- Vertex too small - test1.png, test2.png
- Vertex too large - test3.png, test4.png
- Vertex wrong color - test5.png
- Vertex task in wrong position - test6.png, test7.png, test9.png
- Vertex imput in wrong position
- Vertices too close to each other - test8.png
- Vertices overlapping - test8.png
- Vertices offscreen (completely)
- Vertices offscreen (partly)
- Vertex without parent element
- Vertex displayed incorrectly
- **Vertex in visualization, but not in model**
- Vertex in model, but not in visualization

- Edge too thin
- Edge too wide
- Edge too long
- Edge too short
- Edge wrong color
- Edge not connected to anything
- Edge connecting wrong elements
- Edges too close to each other
- Edges forming ambigous intersections
- Edges offscreen
- Edge in visualization, but not in model
- Edge in model, but not in visualization

- Text too small
- Text too large
- Text wrong color
- Text in wrong position relative to parent element
- Text without parent element
- Texts too close to wach other
- Texts overlapping
- Text overlapping with other elements
- Text offscreen
- Text in visualization, but not in model
- Text in model, but not in visualization
- Text wrong in visualization

## Testcases needed to cover all error types

Each Testcase in all three layers

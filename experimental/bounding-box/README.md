# Draw bounding boxes

This program draws bounding boxes of yolo-annotation format.

## Yolo Annotation Format

Yolo annotation file is space-separated values.
Each line has class-id, boundbox-ceter-position and boundbox-size.
The range for both position and size is between 0 and 1.

```
classid x y width height
...
```

## Class file

Labels corresponding to classid are written line by line

```
person
bicycle
car
motorbike
...
```

## Run command

```sh
$ cabal run bounding-box "label file" "annotation file" "input image file" "output image file"
```

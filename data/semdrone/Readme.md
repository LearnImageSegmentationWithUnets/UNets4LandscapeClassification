# Semantic Drone Dataset

The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts  more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000x4000px (24Mpx). The training set contains 400 publicly available images and the test set is made up of 200 private images.

## Semantic Annotation

The images are labeled densely using polygons and contain the following 22 classes:
  
  - unlabeled
  - paved-area
  - dirt
  - grass
  - gravel
  - water
  - rocks
  - pool
  - vegetation
  - roof
  - wall
  - window
  - door
  - fence
  - fence-pole
  - person
  - dog
  - car
  - bicycle
  - tree
  - bald-tree
  - ar-marker
  - obstacle

## Included Data

* 400 training images
* Dense semantic annotations in png format can be 
    found in "training_set/gt/semantic/label_images/"
* Dense semantic annotations as LabelMe xml files can be 
    found in "training_set/gt/semantic/label_me_xml/"
* Semantic class definition can be 
    found in "training_set/gt/semantic/class_dict.csv" 
* Bounding boxes of persons as LabelMe xml files
    found "in training_set/gt/bounding_box/label_me_xml"
* Bounding boxes of persons as mask images
    found in "training_set/gt/bounding_box/masks"
* Bounding boxes of individual persons as mask images
    found in "training_set/gt/bounding_box/masks_instances"
* Bounding boxes of persons as python pickle file
    found in "training_set/gt/bounding_box/bounding_boxes/person/"

## Contact

aerial@icg.tugraz.at

## Citation

If you use this dataset in your research, please cite the following URL:

www.dronedataset.icg.tugraz.at

## License

The Drone Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:

* That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, we (Graz University of Technology) do not accept any responsibility for errors or omissions.
* That you include a reference to the Semantic Drone Dataset in any work that makes use of the dataset. For research papers or other media link to the Semantic Drone Dataset webpage.
* That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
* That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
* That all rights not expressly granted to you are reserved by us (Graz University of Technology).

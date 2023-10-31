# Craters in Historical Aerial Images (CHAI) Dataset

<p float="center">
  <a href="https://cvl.tuwien.ac.at/" target="_blank">
    <img src="/logos/cvl_white.png" height="75"/>
    &nbsp;&nbsp;&nbsp;&nbsp;
  </a>
  <a href="https://www.tuwien.at/" target="_blank">
    <img src="/logos/tuw_white.png" height="75" />
    &nbsp;&nbsp;&nbsp;&nbsp;
  </a>
  <a href="https://www.luftbilddatenbank-gmbh.at/" target="_blank">
    <img src="/logos/logo_at.png" height="75" />
  </a>
</p>

**PRELIMINARY**, dataset will be published soon.

There are three versions of the dataset: **CHAI-raw**, **CHAI-full**, and **CHAI-light**. The CHAI-raw contains the original 99 historical aerial images with a Region of Interest (ROI) mask as well as the crater annotations. CHAI-full and CHAI-light are the derived datasets that were used for the evaluation of the paper **"CHAI: Craters in Historical Aerial Images"** presented at [WACV2024](https://wacv2024.thecvf.com/). For both datasets we extracted 960×960 patches with an overlap of 20%, both come with the images in .png format and the train, val, and test as .json files in the COCO style. The difference between the CHAI-full and CHAI-light datasets is that for the light version, all patches without any annotations have been removed, which results in the same amount of annotations, but fewer patches:

<table border="1">
  <tr>
    <th rowspan="2">Dataset</th>
    <th colspan="3">Full dataset</th>
    <th colspan="3">Light dataset</th>
  </tr>
  <tr>
    <th>Train</th>
    <th>Val</th>
    <th>Test</th>
    <th>Train</th>
    <th>Val</th>
    <th>Test</th>
  </tr>
  <tr>
    <td>Craters</td>
    <td>22,798</td>
    <td>1,933</td>
    <td>4,458</td>
    <td>22,798</td>
    <td>1,933</td>
    <td>4,458</td>
  </tr>
  <tr>
    <td>Patches</td>
    <td>12,145</td>
    <td>1,173</td>
    <td>2,161</td>
    <td>3,748</td>
    <td>461</td>
    <td>990</td>
  </tr>
</table>

An example of how we split the images can be seen here:

![Overview of the locations contained in this dataset.](/assets/extraction.png)

An overview of the geographical distribution of the images within our dataset in Austria and Germany can be seen here:

![Overview of the locations contained in this dataset.](/assets/location_map.png)

Graz is used in the training dataset, while Vienna and Linz are part of the validation data and all locations in Germany are used for testing.

The images within this dataset are sourced from our industry partner, who gathers them from various outlets, predominantly national archives. In essence, the process of procuring these images can be broken down into three key steps. First, a contractor defines the area of interest to investigate for a particular construction process. Second, preview images from the approximate region are ordered. These preliminary images are typically low-resolution scans of microfilms.
![Overview of the locations contained in this dataset.](/assets/microfilm.jpg)

These microfilms serve as a reference for domain experts to evaluate aspects like image quality – for instance, identifying extensive cloud cover in the top section of the rightmost image – or to determine if an image falls within the region of interest. However, due to the extremely limited resolution of these images, intricate identification of war-related elements such as bomb craters is unfeasible. Consequently, following the assessment, high-resolution scans are requested. 99 of these high-resolution scans make up the CHAI dataset.

## References

TBA

## Download

TBA

## Additional details

TBA

## License

TBA

## Acknowledgements


This work was supported by the Austrian Research Promotion Agency (FFG) under project grant 880883. Acquisition of historical aerial imagery: Luftbilddatenbank Dr. Carls GmbH; Sources of historical aerial imagery: National Archives and Records Administration (Washington, D.C.) and Historic Environment Scotland (Edinburgh).

<p float="center">
  <a href="https://www.bmk.gv.at/" target="_blank">
    <img src="/logos/bmk.png" height="75"/>
    &nbsp;&nbsp;&nbsp;&nbsp;
  </a>
  <a href="https://www.ffg.at/en" target="_blank">
    <img src="/logos/ffg.png" height="75" />
  </a>
</p>

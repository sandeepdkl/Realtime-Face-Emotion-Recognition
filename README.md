# Realtime-Face-Emotion-Recognition

Realtime face recognition using transfer learning.
Here, I have applied MobileNetV2 to do prediction. My dataset contains 7 folders in which each of them contain images of faces of 7 differenet emotions:
angry, disgust, fear, happy, neutral, sad and surprise. I have used this for training purpose. And the testing data contains the 7 folders with same emotions.

There are various types of  transfer learning model for image classification such as: 

1) Xception

2) VGG16

3) VGG19

4) ResNet50

5) InceptionV3

6) InceptionResnet

7) MobileNet

8) DenseNet

9) NASNet

10) MobileNetV2



Here, We are going to use MobileNetV2 to train the model. MobileNetV2 is a pre-trained model.

# 1)What is the Pre-trained Model?

A pre-trained model has been previously trained on a dataset and contains the weights and biases that represent 
the features of whichever dataset it was trained on. Learned features are often transferable to different data. 
For example, a model trained on a large dataset of bird images will contain learned features like edges or 
horizontal lines that you would be transferable to your dataset.


# 2)Why use a Pre-trained Model?

Pre-trained models are beneficial to us for many reasons. By using a pre-trained model you are saving time. 
Someone else has already spent the time and compute resources to learn a lot of features and your model will get benefit from these pre-trained weights.

# 3) MobileNetV2:-

MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the 
residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of 
non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.
MobileNet V2 model has 53 convolution layers and 1 AvgPool with nearly 350 GFLOP. It has two main components:

Inverted Residual Block
Bottleneck Residual Block
There are two types of Convolution layers in MobileNet V2 architecture:

1x1 Convolution
3x3 Depthwise Convolution
These are the two different components in MobileNet V2 model:

# 4) The architecture of MobileNetV2:-

![](https://iq.opengenus.org/content/images/2020/11/conv_mobilenet_v2.jpg)

# 5) Layers in MobileNetV2
<table>
<thead>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>#</td>
<td>Op</td>
<td>Expansion</td>
<td>Repeat</td>
</tr>
<tr>
<td>1</td>
<td>Convolution</td>
<td>-</td>
<td>1</td>
</tr>
<tr>
<td>2</td>
<td>Bottleneck</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>3</td>
<td>Bottleneck</td>
<td>6</td>
<td>2</td>
</tr>
<tr>
<td>4</td>
<td>Bottleneck</td>
<td>6</td>
<td>3</td>
</tr>
<tr>
<td>5</td>
<td>Bottleneck</td>
<td>6</td>
<td>4</td>
</tr>
<tr>
<td>6</td>
<td>Bottleneck</td>
<td>6</td>
<td>3</td>
</tr>
<tr>
<td>7</td>
<td>Bottleneck</td>
<td>6</td>
<td>3</td>
</tr>
<tr>
<td>8</td>
<td>Bottleneck</td>
<td>6</td>
<td>1</td>
</tr>
<tr>
<td>9</td>
<td>Convolution</td>
<td>-</td>
<td>1</td>
</tr>
<tr>
<td>10</td>
<td>AvgPool</td>
<td>-</td>
<td>1</td>
</tr>
<tr>
<td>11</td>
<td>Convolution</td>
<td>-</td>
<td>1</td>
</tr>
</tbody>
</table>

# 6) Convolutions in MobileNetV2
Following is the list of the 53 Convolution layers in MobileNetV2 architecture with details of different parameters like Input height, Input width, Kernel height 
and more:
<table>
<thead>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td># Conv</td>
<td>Input H/W</td>
<td>Input C</td>
<td>Kernel H/W</td>
<td>Stride H/W</td>
<td>Padding H/W</td>
<td>Output H/W</td>
<td>Output C</td>
</tr>
<tr>
<td>1</td>
<td>224</td>
<td>3</td>
<td>3</td>
<td>2</td>
<td>0</td>
<td>112</td>
<td>32</td>
</tr>
<tr>
<td>2</td>
<td>112</td>
<td>32</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>112</td>
<td>32</td>
</tr>
<tr>
<td>3</td>
<td>112</td>
<td>32</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>112</td>
<td>16</td>
</tr>
<tr>
<td>4</td>
<td>112</td>
<td>16</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>112</td>
<td>96</td>
</tr>
<tr>
<td>5</td>
<td>112</td>
<td>96</td>
<td>3</td>
<td>2</td>
<td>0</td>
<td>56</td>
<td>96</td>
</tr>
<tr>
<td>6</td>
<td>56</td>
<td>96</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>56</td>
<td>24</td>
</tr>
<tr>
<td>7</td>
<td>56</td>
<td>24</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>56</td>
<td>144</td>
</tr>
<tr>
<td>8</td>
<td>56</td>
<td>144</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>56</td>
<td>144</td>
</tr>
<tr>
<td>9</td>
<td>56</td>
<td>144</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>56</td>
<td>24</td>
</tr>
<tr>
<td>10</td>
<td>56</td>
<td>24</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>56</td>
<td>144</td>
</tr>
<tr>
<td>11</td>
<td>56</td>
<td>144</td>
<td>3</td>
<td>2</td>
<td>0</td>
<td>28</td>
<td>144</td>
</tr>
<tr>
<td>12</td>
<td>28</td>
<td>144</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>28</td>
<td>32</td>
</tr>
<tr>
<td>13</td>
<td>28</td>
<td>32</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>28</td>
<td>192</td>
</tr>
<tr>
<td>14</td>
<td>28</td>
<td>192</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>28</td>
<td>192</td>
</tr>
<tr>
<td>15</td>
<td>28</td>
<td>192</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>28</td>
<td>32</td>
</tr>
<tr>
<td>16</td>
<td>28</td>
<td>32</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>28</td>
<td>192</td>
</tr>
<tr>
<td>17</td>
<td>28</td>
<td>192</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>28</td>
<td>192</td>
</tr>
<tr>
<td>18</td>
<td>28</td>
<td>192</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>28</td>
<td>32</td>
</tr>
<tr>
<td>19</td>
<td>28</td>
<td>32</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>28</td>
<td>192</td>
</tr>
<tr>
<td>20</td>
<td>28</td>
<td>192</td>
<td>3</td>
<td>2</td>
<td>0</td>
<td>14</td>
<td>192</td>
</tr>
<tr>
<td>21</td>
<td>14</td>
<td>192</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>64</td>
</tr>
<tr>
<td>22</td>
<td>14</td>
<td>64</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>23</td>
<td>14</td>
<td>384</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>24</td>
<td>14</td>
<td>384</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>64</td>
</tr>
<tr>
<td>25</td>
<td>14</td>
<td>64</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>26</td>
<td>14</td>
<td>384</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>27</td>
<td>14</td>
<td>384</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>64</td>
</tr>
<tr>
<td>28</td>
<td>14</td>
<td>64</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>29</td>
<td>14</td>
<td>384</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>30</td>
<td>14</td>
<td>384</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>64</td>
</tr>
<tr>
<td>31</td>
<td>14</td>
<td>64</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>32</td>
<td>14</td>
<td>384</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>14</td>
<td>384</td>
</tr>
<tr>
<td>33</td>
<td>14</td>
<td>384</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>96</td>
</tr>
<tr>
<td>34</td>
<td>14</td>
<td>96</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>576</td>
</tr>
<tr>
<td>35</td>
<td>14</td>
<td>576</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>14</td>
<td>576</td>
</tr>
<tr>
<td>36</td>
<td>14</td>
<td>576</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>96</td>
</tr>
<tr>
<td>37</td>
<td>14</td>
<td>96</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>576</td>
</tr>
<tr>
<td>38</td>
<td>14</td>
<td>576</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>14</td>
<td>576</td>
</tr>
<tr>
<td>39</td>
<td>14</td>
<td>576</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>96</td>
</tr>
<tr>
<td>40</td>
<td>14</td>
<td>96</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>14</td>
<td>576</td>
</tr>
<tr>
<td>41</td>
<td>14</td>
<td>576</td>
<td>3</td>
<td>2</td>
<td>0</td>
<td>7</td>
<td>576</td>
</tr>
<tr>
<td>42</td>
<td>7</td>
<td>576</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>160</td>
</tr>
<tr>
<td>43</td>
<td>7</td>
<td>160</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>960</td>
</tr>
<tr>
<td>44</td>
<td>7</td>
<td>960</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>960</td>
</tr>
<tr>
<td>45</td>
<td>7</td>
<td>960</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>160</td>
</tr>
<tr>
<td>46</td>
<td>7</td>
<td>160</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>960</td>
</tr>
<tr>
<td>47</td>
<td>7</td>
<td>960</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>960</td>
</tr>
<tr>
<td>48</td>
<td>7</td>
<td>960</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>160</td>
</tr>
<tr>
<td>49</td>
<td>7</td>
<td>160</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>960</td>
</tr>
<tr>
<td>50</td>
<td>7</td>
<td>960</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>960</td>
</tr>
<tr>
<td>51</td>
<td>7</td>
<td>960</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>320</td>
</tr>
<tr>
<td>52</td>
<td>7</td>
<td>320</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
<td>1280</td>
</tr>
<tr>
<td>53</td>
<td>1</td>
<td>1280</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>1001</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

# The parameters of each Convolution layer in order are:
<ul>
<li>Input Height and width</li>
<li>Input Channel</li>
<li>Kernel Height and Width</li>
<li>Stride Height/ Width</li>
<li>Padding Height/ Width</li>
<li>Output Height/ Width</li>
<li>Output Channel</li>
</ul>

#This is how the architecture of the model looks like after adding few layers to the already trained model.
![](https://i.ibb.co/NCdhdPw/Model-final-layers.png)

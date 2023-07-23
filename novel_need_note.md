

# Dataset

## Generate new novel need

```
Please provide 10 novel needs for computers in written form. A novel need is defined as the introduction of new terms, features, or the integration of functionalities from other products into computers.
```



## Add noise

* TODO: negative review, slang, emoji

```python
Given some example data of reviews:
1. Thin bezels was an option that was great.
2. A lot of flashy annoying stickers on the chassis.
3. MSI's customer support is a joke too. I submitted several tickets with little to no effect.
4. I wish the fingerprint scanner were a tad bit more sensitive
5.(Beautiful): I love me some apple products! Absoutely beautiful gold. It’s brand new, sealed. Came very quick and in perfect condition. Set up is quick, the Touch ID is super fast. My iPad Pro has the M1 and let’s just say, it’s a game changer. Highly recommend getting this.


# prompt1: Rewrite the following sentences into a review format without changing the meaning of the sentences:

# prompt2: Without changing the main meaning of the sentence, rewrite the following sentences into review forms by adding content, changing sentence structure, etc: 

# prompt3: The following sentences are new user requirements for the product. Rewrite the sentences into the form of review data, and include the new requirements in the generated review data:

1. Integration of artificial intelligence (AI) to assist users in tasks such as organizing files, prioritizing emails, and suggesting optimized workflows for enhanced productivity.
2. Computers should feature AR displays, enabling users to overlay virtual elements onto the real world, enhancing productivity, gaming experiences, and visualizing data in new ways.
3. Computers should possess quantum computing capabilities, enabling them to solve complex problems exponentially faster, revolutionizing fields such as cryptography, optimization, and scientific simulations
4. Computers should seamlessly integrate with smart home devices, allowing users to control and monitor their connected appliances, lights, and security systems directly from their computers.
5. Computers should support wireless charging for peripherals and mobile devices, eliminating the need for cables and providing convenient power transfer.

```

```markdown
1. Integration of artificial intelligence (AI) to assist users in tasks such as organizing files, prioritizing emails, and suggesting optimized workflows for enhanced productivity.

prompt1:
  I'm excited by the prospect of artificial intelligence (AI) integration in our computers. Imagine a computer that assists you in tasks such as organizing files, prioritizing emails, and suggesting optimized workflows for enhanced productivity. The potential for a boost in efficiency is truly remarkable.
  
prompt2:
  The integration of AI in computers has truly revolutionized my workflow. It's as if I have a digital assistant right at my fingertips, helping me organize files, prioritize my emails, and even suggesting optimized workflows for increased productivity. This AI integration has made me more efficient and productive than ever.
  
prompt3:
  (AI Assistant): I've been using this computer for a while now and let me tell you, the integration of artificial intelligence is phenomenal. It assists me in tasks like organizing my files, prioritizing my emails, and even suggesting optimized workflows which have significantly enhanced my productivity. A game changer for sure!

2. Computers should feature AR displays, enabling users to overlay virtual elements onto the real world, enhancing productivity, gaming experiences, and visualizing data in new ways.

prompt1:
AR displays on computers would be a game changer. This would enable users to overlay virtual elements onto the real world, enhancing productivity, gaming experiences, and visualizing data in new ways. The ability to interact with data and digital objects in a physical space would take our computing experiences to the next level.

prompt2:
The introduction of AR displays in computing has opened up a whole new world for me. Now, I can overlay virtual elements onto the real world, enhancing not just my productivity but also my gaming experiences. Visualizing data has never been this interactive and intuitive. This is a game-changer in the field of computing!

prompt3:
(AR Enthusiast): This computer is amazing. One of its key features is the AR display that overlays virtual elements onto the real world, which has truly changed my gaming experience. It also enhances my productivity and helps visualize data in a whole new way. A truly immersive experience.

3. It would be fantastic if the laptop has a projector.
prompt1:
A laptop equipped with a projector would be an incredible asset. This feature would transform the laptop into a mobile presentation tool, enhancing its versatility.

prompt2:
It would be absolutely fantastic if a laptop came with a built-in projector. This feature would allow for on-the-go presentations and could transform any space into a mini theater. It's a feature that I believe could revolutionize the way we view media and share content on our laptops.

prompt3:
(A True All-in-One Device): Imagine my delight when I discovered that this laptop also doubles as a projector! It's truly fantastic to have such versatility in a single device.

```



## New Generation V1

* Prompt

```markdown
You need to help me generate some reviews about computers. The content of these texts needs to look very real, be practical, and at the same time, imaginative. The overall text should include any content that might appear in real reviews, such as evaluations of the computer's existing functions, price, logistics and after-sales service, and expectations for the computer, expressing hopes for what the computer can do but is currently unable to do. The reviews can be positive, neutral or negative, but they must all contain imaginative new requirements.
Specific examples are as follows:

1. I think this computer is overall not bad, and the cooling is okay, but I hope it can add a voice assistant that can help me automatically find files and clean up junk, which would be more convenient.

2. The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.
Please generate 10 reviews that meet the above requirements.


```



```markdown
Please provide an additional 10 review texts. They should feature a richer sentence structure, include more realistic appraisal content, and contain more imaginative new requirements within the content.
```



```python
Please provide an additional 10 longer review texts. They should feature a richer sentence structure, include more realistic appraisal content, words expressing complaints and emotions, or meaningless content., and contain more imaginative new requirements within the content.
```



**Example**

```python
1. I think this computer is overall not bad, and the cooling is okay, but I hope it can add a voice assistant that can help me automatically find files and clean up junk, which would be more convenient.
2. The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.
3. This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.
4. The graphics card on this machine is top-notch, excellent for gaming. However, the speakers could use a boost. It would be fantastic if they added surround sound capabilities, to immerse you in the gaming experience fully.
5. The precision of the touch screen on this computer is quite good, but I can't help but wish for a 3D holographic display that I could manipulate with hand gestures, like in those sci-fi movies.
6. I appreciate the high-quality audio of the speakers on this computer, but the lack of noise cancellation can be a downside during video calls. A futuristic feature would be an AI-powered noise cancellation system that can differentiate between voice and background noise.
```





* train
  * Novel_need: 63
  * Others: 13256
* test
  * novel_need: 17
  * Others: 3313

#### Result

* pre: 1
* rec: 1
* f1: 1

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 17        | 0         |
| Label=0 | 0         | 3313      |

##### Top K

|                   | Precision | Recall | F1     |
| ----------------- | --------- | ------ | ------ |
| Top 5             | 1         | 0.2941 | 0.4545 |
| Top 10            | 1         | 0.5882 | 0.7407 |
| Top 15            | 1         | 0.8824 | 0.9375 |
| Top #true label=1 | 1         | 1      | 1      |



#### Visualization

<img src="./novel_need_note.assets/image-20230716173211420.png" alt="image-20230716173211420" style="zoom:50%;" />

* Fine-tuned bert

### <img src="./novel_need_note.assets/image-20230716173221028.png" alt="image-20230716173221028" style="zoom:50%;" />



# Find novel need from review

* train
  * Novel_need: 22
  * Others: 397
* test
  * novel_need: 8
  * Others: 97

### Experiment

#### V1

```sh
python run_glue_no_trainer_v1.py \
  --model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir tmp \
  --train_file data/novel_train_new.csv \
  --validation_file data/novel_valid_new.csv
```

* Pre: 0.46
* Rec: 0.5
* F1: 0.48

#### V2 

train:

* 1: 68
* 0: 1704

test:

* 1: 12
* 0: 431

```sh
python run_glue_no_trainer_v2.py \
  --model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir tmp \
  --train_file data/novel_train_v2.csv \
  --test_file data/novel_valid_v2.csv
```

* pre: 0.8824
* Rec: 0.8824
* f1: 0.8824

![image-20230616172922906](./novel_need_note.assets/image-20230616172922906.png)

#### V3

train:

* 1: 66
* 0: 13427

test:

* 1: 14
* 0: 3360

```sh
python run_glue_no_trainer_v2.py \
  --model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir tmp \
  --train_file data/novel_train_v3.csv \
  --test_file data/novel_valid_v3.csv
```

* pre: 0.9333
* rec: 1
* f1: 0.9655

![image-20230616172910611](./novel_need_note.assets/image-20230616172910611.png)



![image-20230616172944342](./novel_need_note.assets/image-20230616172944342.png)

#### Exp-Prompt1

```shell
python run_glue_no_trainer_v2.py \
  --model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir tmp \
  --train_file data/novel_train_prompt2.csv \
  --test_file data/novel_valid_prompt2.csv
```

train:

* 1: 63
* 0: 13430

test:

* 1: 17
* 0: 3357

**Result**

* pre: 0.8846
* rec: 0.9583
* f1: 0.92

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 23        | 1         |
| Label=0 | 3         | 3303      |

##### Top K

|                   | Precision | Recall | F1     |
| ----------------- | --------- | ------ | ------ |
| Top 5             | 1         | 0.2083 | 0.3445 |
| Top 10            | 1         | 0.4167 | 0.5882 |
| Top 15            | 1         | 0.625  | 0.7692 |
| Top #true label=1 | 0.9167    | 0.9167 | 0.9167 |



#### Exp-Prompt2

train:

* 1: 63
* 0: 13430

test:

* 1: 17
* 0: 3357

**Result**

* pre: 0.9444
* rec: 1.0
* f1: 0.9714

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 17        | 0         |
| Label=0 | 1         | 3556      |









### TODO

* ~~instruction tuning~~
* ~~In-context learning：existing needs~~
* ~~chain-of-thought~~
  * ~~e.g. 解释novel need: needs组合~~



* two-stage classifier
  * gpt生成irrelevant sentence
* one-stage
  * testing+noise: similar to review data

### 指标

Precision @ K

**Recall @ K** 

* data -> model -> [0: prob, 1:prob] -> select prob of label 1 -> sort -> top@k

Ranking

...



# LLM

## model

* GPT3/3.5
* ChatGLM-6B
  * https://github.com/THUDM/ChatGLM-6B

## Instruction tuning

```python
Given some examples:
novel need: I needed something small for work and travel, but strong enough to do light gaming, preferably heavy gaming.
novel need: I wanted a powerhouse that was light and fast.
not novel need: Two months later it doesn't work.
not novel need: I know you get what you pay for with features and such, but this machine sounded a little bit like one of those awful leaf-blowers with intermittent revving if I did so much as open windows explorer.
not novel need: If you’re wanting a laptop for word processing, web surfing, movies and occasional gaming, this truly is the best bang for your buck.
novel need: I wish on the PSU there would have been a USB port.

Find the label (novel need or not novel need) of the sentence:
I've yet to update it from whatever version came with it due to a problem with storage even though I put an SD card in
```



## Instruction Tuning V2

* **Prompt**

```python
Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements).
Here are some examples:

The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.	1
This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.	1
The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience	0
i had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.	0

Given the following reviews, please return the judgement result:
```

* **Model**: **chatgpt**, **chatglm-6b**

### case study (label=1)

* I adore  the fast boot time on this machine, but the customer service was  underwhelming when I had an issue. If only the computer could have a built-in  diagnostic tool that could troubleshoot and fix problems automatically!
  * **chatgpt** Return: "1"
  * **chatglm-6b** Return: "Based on the given reviews, the review data does not contain new requirements. The computer currently has fast boot time and customer service is mentioned as good. The only new requirement mentioned in the review is the desire for a built-in diagnostic tool."
* The  display on this computer is stunning, but the touchpad feels a bit outdated.  I'd love it if they could integrate gesture recognition technology for a  seamless and intuitive navigation experience.
  * **chatgpt** Return: "1"
  * **chatglm-6b** Return: "Based on the given reviews, the review data contains new requirements. The reviewer mentions the desire for gesture recognition technology in the touchpad, which is a feature that is not currently available on the computer."
* I'm  impressed with the durability of this computer, but the noise of the fan can  be a bit distracting. I wish they would invent a silent cooling system that  could keep the computer cool without a sound.
  * **chatgpt** Return: "1"
  * **chatglm-6b** Return: "Based on the given reviews, the review data does not contain new requirements. The computer already has a silent cooling system, but the reviewer is wishes it had a silent cooling system that could keep the computer cool without a sound."
* The  keyboard layout of this computer is pretty intuitive, but the lack of  backlighting makes it hard to use in dimly lit conditions. A feature I would  love to see is smart keyboard lighting that adjusts based on the room's light conditions.
  * **chatgpt** Return: "1"
  * **chatglm-6b** Return: "1. Yes, the review data contains new requirements for the computer's keyboard lighting feature."
* This  computer's large storage capacity and fast read-write speeds are truly  commendable, making file transfers a breeze. Unfortunately, the sound system  does not match up. The speakers are tinny and lack depth. If they were to  introduce a feature that creates a virtual surround sound experience, it  would be a massive leap forward in enhancing multimedia consumption.
  * **chatgpt** Return "1"
  * **chatglm-6b** Return: "1 (contains new requirements)"

### case study (label=0)

* I love my Acer  Chromebook. I bought it for online college and gaming. The color is so vivid  and perfect; the picture crystal clear and the sound is Dolby Sound. I am  extremely happy with this product and the cost was only $229.00.

  * **chatgpt** Return: "0"
  * **chatglm-6b** Return: "1. Yes, the review data contains new requirements. The reviewer mentions the lack of cooling and the potential for loud/hot operation, which are features that are not currently available on the Acer Chromebook."

* Absolutely  amazing laptop! Runs games with zero lag. I was lucky to get it for $579, and  would recommend checking this frequently as it goes super fast at that price.

  * **chatgpt** Return: "0"
  * **chatglm-6b** Return: "1. New requirements: The review mentions a lack of cooling and the use of a freeze-like technology, which suggests that the computer may not be designed to handle high- demanding games. These features are not currently available on the laptop."

* Simple, this  laptop is great for the value. (1,399 plus I had the option to pay in 5  installments was a huge deal maker!!!) This would be my second MSI laptop in  5 years (my other one still works perfect I just wanted better specs)Good  specs 10th gen i7 with RTX 2070. The SSD is super fast. I also added a  2.5" Samsung evo 1TB SSD on the SATA drive (highly recommend for more  storage).Stays cool enough that I can play for hours (around 80/81 degrees).  and that is without using the cool boost. ( I never even turn that thing  on)The only thing I will complain, but its not a big deal is the screen can  be better ( IPS 3ms, 144hz) and the speakers are not the best honestly I  think these are the worst speaker I have ever heard. But on the other side I  never need them anyways since I use a headset so not a big deal. What you are  paying for this awesome powerhouse speakers should not be a deal breaker.The  construction is hard plastic, but it does feel of good quality ( I come from  a raider series with aluminum construction). It is also not very heavy,  coming in at 15" screen it is really easy to move around and take  anywhere. Battery life is about 2 hours (it does have a big graphics card and  rgb lighting, sooooo its going to eat it up fast) overall I would recommend  this laptop for any amateur gamer or even for work that requires graphics and  a lot of processing.

  * **chatgpt** Return: "1"

  * **chatglm-6b** Return: "Based on the reviews provided, the review data contains new requirements. The reviewer mentions the need for a more efficient cooling system and the improvement of the screen quality, which are features that the current laptop does not have. Additionally, the reviewer mentions the need for a larger storage capacity and the improvement of the speaker quality, which are features that the current laptop does not have. These new requirements indicate that the computer does not currently have these features, and the reviewer would like to see them added.

    Therefore, the judgement result is 1."

* Parents bought  this for my sister and this is perfect for her to start on it's good for  light-medium gaming. With school converting to online it's perfect as well.  This computer also has good battery life away from the charger while might  use with heavy use not so much it's good though. Overall the computer is good  for it's current price and an amazing deal.

  * **chatgpt** Return: "0"
  * **chatglm-6b** Return: "Based on the given reviews, the computer does not contain new requirements."

* as much as I like  this laptop at one point I started getting bluescreens telling me something  was wrong with it. I resolved the issue but it caused many problems during my  homework.

  * **chatgpt** Return: "1"
  * **chatglm-6b** Return: "Based on the review, it does not contain new requirements. The reviewer is discussing a problem with the laptop and its potential impact on their work, but it does not indicate that the computer does not currently have the features that the reviewer is looking for."



## LoRA

<img src="./novel_need_note.assets/image-20230629222313826.png" alt="image-20230629222313826" style="zoom:50%;" />

<img src="./novel_need_note.assets/image-20230629222301551.png" alt="image-20230629222301551" style="zoom:50%;" />

* It freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture
*  LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.

### Model

Chatglm-6B: https://github.com/THUDM/ChatGLM-6B/tree/main

Chatglm-6B Lora finetune: https://github.com/hiyouga/ChatGLM-Efficient-Tuning



### Dataset

* format: json

```python
{"instruction": '''Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements).
Here are some examples:
 
The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.	1
This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.	1
The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience	0
I had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.	0

Given the following reviews, please return the judgement result:''', 
 
 "input": "My daughter loves her new laptop!", 
 "output": "0"}, 
```

* Train:
  * label=1: 56
  * label=0: 3000
* Test:
  * label=1: 24
  * label=0: 300



### Experiment

```python
python src/train_sft.py \
    --do_train \
    --model_name_or_path path_to_model \
    --dataset nn_train \
    --finetuning_type lora \
    --output_dir path_to_output \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --fp16
```

#### Exp1 (epoch=1)

<img src="./novel_need_note.assets/image-20230629231544380.png" alt="image-20230629231544380" style="zoom:50%;" />

* Format is not correct
* Almost 70% predictions are "1"

#### Exp2 (epoch=2)

<img src="./novel_need_note.assets/image-20230629232030618.png" alt="image-20230629232030618" style="zoom:50%;" />

Result after manually processed:

* Precision: 0.28
* Recall: 0.875
* F1: 0.424

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 21        | 3         |
| Label=0 | 54        | 297       |



#### Exp3 (epoch=3)

<img src="./novel_need_note.assets/image-20230629232830785.png" alt="image-20230629232830785" style="zoom:50%;" />

* All predictions are: **"0"**



#### TODO

* Epoch之间 evaluate
* 生成概率-> **top k**

* ~~gpt3.5 api调用chatgpt (概率)~~
* ~~**正样本->生成长的，短的 （分布）**~~
* loRA, pTuning

* Baseline



# Add noise

## 1. ChatGPT generation

```markdown
Your task is to add noise to comments with new requirements. "New requirements" refer to features that computers currently do not have, while "noise" refers to any content that might appear in computer product reviews on e-commerce platforms.

Here are a few regular long comments:

1. This is a good PC for the money. I am predominantly using it for graphic design, and it allows me to use multiple applications at once with no lag. It is additionally good for gaming, as well as business use (it is my business laptop as well). Though built for gamers, this is an all around well made machine. Best part of the PC is it is FAST.I will note the fan is a little loud, but that isn't a surprise based on other reviews. Also, the battery life isn't great, but i keep it plugged in most of the time. Screen resolution and quality is good, though nothing to brag about compared to other laptops in the market today. Using multiple monitors it doesn't bother me much.I would recommend this computer as one of the best bang-for-your-buck purchases you can get in the market today.

2. Bought one and 4 days after receiving it, it looked like first 2 pics upon booting. Replaced it and the last 2 pics are what the replacement looked like after a few days as well. It would not happen every time, but it would have that white line around the edge for maybe 30 min and would fade back to normal. More recently, the replacement won't turn on immediately after hitting power button. The screen stays black and no response for 15 minutes or so. I wanted to give it one more chance, because I haven't read about this happening to anyone else and I received these two laptops right before Amazon went out of stock, so I thought maybe I just got bad ones at the end but, unfortunately, Amazon can only provide a refund and are still out of stock.

3. After sending back my other computer (an Acer) I paid the extra 100$ and got this guy for gaming. After 30 minutes of updates....It blew my mind how awesome this machine is. The keyboard lighting is fully customizable. It has overclock setting built in so you do not have to crash your computer a bunch of times to get it right. I play all my games on Ultra and it is 100fps+. I am so glad I went with the Predator and will be staying with this from now on. Also, I will be installing a 1tb SSD in the empty slot to add more space. Can't say enough good things about how light, stylish and out of the box ready for gaming it is!

4. I use an XPS 17 as my workstation. For going places though, this super small laptop can do everything I need. It has 2 Thunderbolt 4 ports (USB C) and a micro-SD card reader. I use the reader with a 1TB micro-SD to store all my reference files. This makes the 512GB SSD more than adequate. My XPS 13 has 16GB ram which is more than enough for the purposes for which I use this machine. My only wish is if one had a 5G option so I would have to use a Hot Spot connection. The 1920x1200 display resolution is perfect. The XPS 13 is too small to really use 4K properly and battery life on 4K laptops is significantly less than these HD+ machines.

Expand the following comments with new requirements to over 300 words, but the content must retain the new requirements:

The keyboard comes with customizable backlighting and macros, letting you tailor your typing experience to your preferences and needs, in line with the 2018 standard for keyboards.

```

## 2. Manually written

```python
## 京东
品控问题很大，底座不平，就像个电报机，而且电扇太吵太吵了！！能不能加一个完全0噪声的电扇，我需要无声工作环境？此外吐槽下第一次申请售后，工程师在电话中告知未删除密码，需要退回操作，等机器返回后删除密码，我再次申请，售后以检测无故障为理由拒绝售后审核，并且不断让我补充信息，我打了很多次客服电话申请，最后是客服看不下去了才给的二次返回去，这次售后再次给出无故障的检测，并且原返，在线客服告诉去线下自己检测，消费者真是弱势群体啊，这么明显的质量问题难道他们看不见吗？我今天就在这里给各位提个醒，谨慎购买！他们只会来回推诿扯皮！这种问题大家看看视频就能判定的，为什么要不断检测呢？没办法，这条视频我只能放在这里了，而且我会维权到底，我就不信了，我还找不到说理的地方了吗！

质感非常的不错。全金属CNC.给人一种高级感，在高铁上办公也不失风采，这键盘手感有改观，反馈力度不错，打字得心应手！再配上处理器的性能，屏幕的观感，是创造生产力的贤内助，屏幕颗粒感肉眼看比我之前的Acer好的多，LCD的屏幕不那么晃眼，个人认为电脑屏幕还是LCD的要好一些，三档刷新率60hz，90hz，120hz任意切换（fn+K键），插电就高刷，出差有续航焦虑就60。强烈建议可以整个裸眼3D，提高沉浸感。特别是语音助手给人焕然一新的感觉，问个天气，问个问题，马上就告诉你答案，非常好。处理器的调度也比较开放，秒杀独立显卡不是问题。最后希望做得越来越好，创造出更多令人感动的科技产品。

家里的老本坏了，单位两个同事都是用的这款笔记本，最长的一位用了大半年了，两位都说这款用着非常好用，配置高，性价比高，运行也快，都极力推荐这款，所以也入手了这款。入手后也果然没让我失望，全金属外壳，外观时尚，运行很快，开机也非常快，感觉不到十秒就完成了，送的office也正是需要的，很赞！作为一个区块链技术的工作者，希望可以集成个分布式电脑，然后用中枢电脑快速安全地处理保存数据。最后来说说客服，非常专业有耐心，有问题自己解决不了，客服不厌其烦一条一条回复，一点一点教，非常好
```



## 负样本长度统计（抽样）

<img src="./novel_need_note.assets/image-20230707002900142.png" alt="image-20230707002900142" style="zoom:50%;" />

## 正样本（处理前）长度统计

<img src="./novel_need_note.assets/image-20230707002746633.png" alt="image-20230707002746633" style="zoom:50%;" />



## 正样本（处理后）长度统计

<img src="./novel_need_note.assets/image-20230707002806206.png" alt="image-20230707002806206" style="zoom:50%;" />

### Examples (long reviews with novel needs)

```markdown
It's not good at all. The buzzing sound is particularly loud and the customer service is not good. I haven't used it much and the right side of the keyboard can't be used. Facial recognition also can't be used. From the time I got the computer, fingerprint recognition couldn't be used. The customer service just said it's just a design. Am I buying your design or performance? I can't stand it. It's difficult to use. And customer service doesn't care about anything. After buying it, the price dropped. Also, charging is too troublesome. **Can't a wireless charger be added to the computer?** The power cord is too long and takes up space. To tell the truth, with this money, I might as well buy other brands. I suspect that the good reviews are all brushed out, right? The money is used to buy the water army and marketing number, right? The performance is not good to use, it's rubbish.

To be honest, it's quite ordinary. Except for the good appearance, there's nothing else, the running speed is ordinary and not as good as the previous computer, I have paid the IQ tax, I truly regret it. The running speed is very ordinary, it's easy to freeze when opening large Excel or Word files. The screen effect is first-class, no one can beat it. But if you don't have experience, you will not know how to use it, the system is too different from Windows. **Could you have an intelligent assistant for one-to-one teaching, where you don't know how to use it, just automatically open the prompt and such.** Otherwise, it's too hard to learn. The keyboard feels super good, and the touchpad is also incredibly good. Of course, the system doesn't have various junk advertisements, it can be praised for that, I really can't stand the Windows dominated by pop-up windows anymore. The craftsmanship is truly top-notch, the feel is also okay, but my model is a bit heavy, it's not convenient enough to carry. Logistics service gets full marks. Ordered yesterday, it was delivered in two or three days,

This computer really took my attention due to the intricate detailing it allows with the keyboard, something that you don't often find in your typical machines. The keyboard is equipped with fully customizable backlighting and macros, which isn't just a perk but seems to be more of a necessity with the shift towards more personalized technology experiences. These innovative features on the keyboard allow you to tailor your typing experience according to your unique preferences and needs. And it's not just about how it illuminates the keys in a dazzling array of colors for aesthetic purposes or to type in the dark. The customizability goes way beyond that, giving users the chance to truly make their computing experience their own. Each key can be individually programmed to perform specific functions, whether it's launching a particular application, executing a series of commands, or just typing out a frequently used phrase. This feature is indeed in line with the 2018 standard for keyboards but goes well beyond just ticking a box. Imagine, being a graphic designer, where you frequently use a combination of keys to perform certain tasks in design software. With this keyboard, you can set those combinations as macros. All those tasks can be performed with a single keystroke, enhancing your productivity by leaps and bounds. Gamers, too, would find this feature to be an absolute game-changer as it would allow them to program complex in-game maneuvers to individual keys, offering them a significant advantage during gameplay. **Additionally, in the future, I would love to see keyboards that are equipped with touch-sensitive keys, adding a whole new layer of interactivity and functionality.** For instance, the pressure applied to a key could determine the speed of an on-screen character in a game or the thickness of a line in a graphic design tool. A built-in haptic feedback feature would be fantastic too, providing real-time tactile responses to our keystrokes, enhancing the typing experience even further. Although this keyboard is indeed a considerable step up from what we are used to, there's always room for growth, and I am excited about the possibilities for the future.

The old laptop at home broke down. Two colleagues at work use this laptop, the one who has been using it for the longest has been using it for over half a year. Both said that it's very good to use, high configuration, high cost performance, fast operation, and highly recommended this model. So, I also bought this model. After buying it, it didn't disappoint me. Full metal shell, fashionable appearance, fast operation, very fast boot up, it feels like it completes in less than ten seconds, the Office it comes with is just what I need, great! As a blockchain technology worker, **I hope it can integrate a distributed computer and then use the central computer to quickly and securely process and save data.** Finally, talking about customer service, very professional and patient. If you can't solve a problem, the customer service patiently replies one by one, teaches bit by bit, very good.
```





# Instruction Tuning V3

## Prompt

```python
Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements).
Here are some examples:

The computer runs fast, but the cooling is lacking. It would be great if it could use a freeze-like technology to cool down in a second. And then there is the screen quality is not good, cracks appeared without much use.	1
This computer has a great processing speed and the design is sleek, but the battery life leaves a lot to be desired. I hope in the future they could incorporate solar charging features so I could work all day without worrying about running out of power.	1
The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience	0
i had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.	0

Given the following reviews, please return the judgement result:
```

## Data

* Positive: 91
* Negative: 100

### Abnormal return

```python
## label=0
1. 'Based on the given review, "EXCELENTE PRODUCTO, PERFECTO ESTADO - ECUADOR," it does not mention any specific requirements or features that the computer should have. Therefore, the review does not contain new requirements. The judgement result is 0.'
2. '1\n1\n0\n0\n0'
```

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 88        | 3         |
| Label=0 | 36        | 62        |

Precision: 0.7097

Recall: **0.9670**

F1: 0.8186



# TODO

* **prompt bert: top @K (概率)**
* ~~加些短的~~
* ~~1:50， 1:100 样本比例~~
* R drop
  * https://www.msra.cn/zh-cn/news/features/r-drop
* curriculum learning： https://aclanthology.org/2021.naacl-main.434.pdf





# Generate V2 (Bert)

```sh
 python run_glue_no_trainer_v1.py \  
 --model_name_or_path bert-base-uncased \  
 --per_device_train_batch_size 32 \  
 --learning_rate 2e-5 \  
 --num_train_epochs 3 \  
 --output_dir tmp \  
 --train_file data/novel_train_generate_v2.csv \  
 --validation_file data/novel_valid_generate_v2.csv
```

### Dataset

#### Length statistic

* Label=1
  * <img src="./novel_need_note.assets/image-20230716192959226.png" alt="image-20230716192959226" style="zoom:50%;" />
* Label=0
  * <img src="./novel_need_note.assets/image-20230716193026294.png" alt="image-20230716193026294" style="zoom:50%;" />

* train: 
  * 1: 78
  * 0: 13,255
* Test:
  * 1: 20
  * 0: 3314

### **Result**

* pre: 1
* rec: 0.9
* f1: 0.9474

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 18        | 2         |
| Label=0 | 0         | 3314      |

#### Top K

|                     | Precision(tp/k) | Recall(tp/20) | F1     |
| ------------------- | --------------- | ------------- | ------ |
| Top 5               | 1               | 0.25          | 0.4    |
| Top 10              | 1               | 0.5           | 0.67   |
| Top 15              | 1               | 0.75          | 0.8571 |
| Top #(true label=1) | 0.9             | 0.9           | 0.9    |



### Visualization

#### Before Fine-tuned

<img src="./novel_need_note.assets/image-20230716193131395.png" alt="image-20230716193131395" style="zoom:50%;" />

#### After Fine-tuned

<img src="./novel_need_note.assets/image-20230716193150353.png" alt="image-20230716193150353" style="zoom:50%;" />



# Generate V2 (GPT3.5)

## Dataset

Test

* 1: 20
* 0: 100 (sample)

## Exp1

### Prompt

```python
prompt = '''Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements).
Here are two examples:

The pre-installed software on this computer is helpful, but the bloatware is a nuisance. An innovative idea would be an intelligent software manager that can automatically detect and uninstall unwanted programs.-->1
I am bringing my review down to 3 stars for now as I had to just send the computer back to the manufacturer because it wouldn't turn on. I have had the computer less than 2 months and already an issue. Before this happened I did like the computer and loved how fast it loaded and operated as just a basic model laptop. However, not even 2 months later it wont turn on and I had to pay to ship to the manufacture to see what the issue is.I bought this as a replacement to the laptop that I had. The one I had was extremely slow and would have an error and restart on its own every once in awhile. This Chromebook is a 1000x faster than the old laptop. It's great for just a computer to have access to the internet. If you don't need much out of a laptop then this one is great, has a couple USB ports and an HDMI port as well as blue tooth ability. I also like how the screen is less reflective so it has less glare. Overall, great product for the short time I have had it!-->0

Given the following review, please return the judgement result:'''
```

### **Result**

* Pre: 0.303
* Rec: 1
* F1: 0.4651

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 20        | 0         |
| Label=0 | 46        | 51        |

*3 reviews cannot be determined (label=0)*

### False Positive

* Label=0 and Predict=1

  * ```
    What can I say, the laptop is amazing, the screen is 99% RGB, the CPU Ryzen R7 5800h is a monster, the rtx 3050ti with just 4gb of vram is a horrible thing, but still you can't even build a desktop more powerful than this thing for $999, I added 16gb extra ram to a total of 24gb and also a 1tb SSD to a total of 1.5tb SSD this laptop is great for games like red dead redemption 2, GTA v, Doom eternal, but the best thing for me is video editing, it's like 3x faster than my i7 7700k, but this thing is a laptop! You can manually set the tdp limits to make it run really cool and silent when you need it, or Max fans and Max power when you just want UNLIMITED POWAAAAA! 8 cores 16 threads @4.4ghz on a laptop was beyond dreaming just 3 years ago, I really love AMD, the rtx is severely castrated with those 4gb vram but still with dlss and fsr I'm sure it will handle anything for at least 5 years to come
    ```

  * ```
    I am used to a window program... so at first I was uncomfortable with the chromebook. It has basic uses but it is very efficient and serves my work needs. I do not yet regret it.
    ```

  * ```
    it is so much lighter and more portable
    ```

## Exp2

### Prompt

```python
Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements).
Here are few examples:

The pre-installed software on this computer is helpful, but the bloatware is a nuisance. An innovative idea would be an intelligent software manager that can automatically detect and uninstall unwanted programs.-->1
I am bringing my review down to 3 stars for now as I had to just send the computer back to the manufacturer because it wouldn't turn on. I have had the computer less than 2 months and already an issue. Before this happened I did like the computer and loved how fast it loaded and operated as just a basic model laptop. However, not even 2 months later it wont turn on and I had to pay to ship to the manufacture to see what the issue is.I bought this as a replacement to the laptop that I had. The one I had was extremely slow and would have an error and restart on its own every once in awhile. This Chromebook is a 1000x faster than the old laptop. It's great for just a computer to have access to the internet. If you don't need much out of a laptop then this one is great, has a couple USB ports and an HDMI port as well as blue tooth ability. I also like how the screen is less reflective so it has less glare. Overall, great product for the short time I have had it!-->0
I don't usually give reviews, but this little guy is perfect for homework and general use. It's not a super powerful machine, but for the price it is perfect for a homework type computer. I have a Mac for design and web development, but this laptop is great for what I need it for, I would suggest making use of one drive or an external hard drive for storage.-->0
I had to send this back. The keyboard doesn’t work, types BT instead of T and the letter U was messed up as well? Box it came in was pretty battered and the top of the laptop has a dent.-->0
I like this chromebook, it's easy to hook up and because it only runs on chrome doesn't get too complicated. I am 72 yrs old and not really good with technology so it is a bit of learning curve for me. I would certainly recommend it.Edit: I am dissapointed that I can't do video calls on it. The camera only points to me and not the subject. Didn't realize this until recently.-->0

Given the following review, please return the judgement result:
```

### Result

* Pre: 0.2985
* Rec: 1
* F1: 0.4598

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 20        | 0         |
| Label=0 | 47        | 53        |



## Exp3

### Prompt

```python
Given a computer purchase review, determine whether the review contains “novel needs”. A novel need refers to a function not included in the computer. If there exists novel needs, return 1; if not, or if the comment is meaningless, return 0.
Here are few examples:

The pre-installed software on this computer is helpful, but the bloatware is a nuisance. An innovative idea would be an intelligent software manager that can automatically detect and uninstall unwanted programs.-->1
I am bringing my review down to 3 stars for now as I had to just send the computer back to the manufacturer because it wouldn't turn on. I have had the computer less than 2 months and already an issue. Before this happened I did like the computer and loved how fast it loaded and operated as just a basic model laptop. However, not even 2 months later it wont turn on and I had to pay to ship to the manufacture to see what the issue is.I bought this as a replacement to the laptop that I had. The one I had was extremely slow and would have an error and restart on its own every once in awhile. This Chromebook is a 1000x faster than the old laptop. It's great for just a computer to have access to the internet. If you don't need much out of a laptop then this one is great, has a couple USB ports and an HDMI port as well as blue tooth ability. I also like how the screen is less reflective so it has less glare. Overall, great product for the short time I have had it!-->0

Given the following review, please return the judgement result:
```

### Result

* Pre: 0.3279
* Rec: 1
* F1: 0.4939

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 20        | 0         |
| Label=0 | 41        | 58        |

*1 reviews cannot be determined (label=0)*



## Exp4

### Prompt

```python
Given a computer purchase review, determine whether the review contains “novel needs”. A novel need refers to a function not included in the computer. If there exists novel needs, return 1; if not, or if the comment is meaningless, return 0.
Here are few examples:

The pre-installed software on this computer is helpful, but the bloatware is a nuisance. An innovative idea would be an intelligent software manager that can automatically detect and uninstall unwanted programs.-->1
I am bringing my review down to 3 stars for now as I had to just send the computer back to the manufacturer because it wouldn't turn on. I have had the computer less than 2 months and already an issue. Before this happened I did like the computer and loved how fast it loaded and operated as just a basic model laptop. However, not even 2 months later it wont turn on and I had to pay to ship to the manufacture to see what the issue is.I bought this as a replacement to the laptop that I had. The one I had was extremely slow and would have an error and restart on its own every once in awhile. This Chromebook is a 1000x faster than the old laptop. It's great for just a computer to have access to the internet. If you don't need much out of a laptop then this one is great, has a couple USB ports and an HDMI port as well as blue tooth ability. I also like how the screen is less reflective so it has less glare. Overall, great product for the short time I have had it!-->0

You may have misjudged these examples, their labels are 0, not 1:
I like this chromebook, it's easy to hook up and because it only runs on chrome doesn't get too complicated. I am 72 yrs old and not really good with technology so it is a bit of learning curve for me. I would certainly recommend it.Edit: I am dissapointed that I can't do video calls on it. The camera only points to me and not the subject. Didn't realize this until recently.
The MacBook delivers top-notch performance! It’s 14 hours of battery life delivers!!! The bright and colorful display all in is a delightfully portable design. It’s truly worthy of the Air name.

Given the following review, please return the judgement result:
```

### Result

* Pre: 0.3846
* Rec: 1
* F1: 0.5555

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 20        | 0         |
| Label=0 | 32        | 68        |



## Exp5

### Prompt

```python
Given a computer purchase review, determine whether the review contains “novel needs”. A novel need refers to a function not included in the computer. If there exists novel needs, return 1; if not, or if the comment is meaningless, return 0.
Here are few examples:

The pre-installed software on this computer is helpful, but the bloatware is a nuisance. An innovative idea would be an intelligent software manager that can automatically detect and uninstall unwanted programs.-->1
I am bringing my review down to 3 stars for now as I had to just send the computer back to the manufacturer because it wouldn't turn on. I have had the computer less than 2 months and already an issue. Before this happened I did like the computer and loved how fast it loaded and operated as just a basic model laptop. However, not even 2 months later it wont turn on and I had to pay to ship to the manufacture to see what the issue is.I bought this as a replacement to the laptop that I had. The one I had was extremely slow and would have an error and restart on its own every once in awhile. This Chromebook is a 1000x faster than the old laptop. It's great for just a computer to have access to the internet. If you don't need much out of a laptop then this one is great, has a couple USB ports and an HDMI port as well as blue tooth ability. I also like how the screen is less reflective so it has less glare. Overall, great product for the short time I have had it!-->0

You may have misjudged these examples, their labels are 0, not 1:
I like this chromebook, it's easy to hook up and because it only runs on chrome doesn't get too complicated. I am 72 yrs old and not really good with technology so it is a bit of learning curve for me. I would certainly recommend it.Edit: I am dissapointed that I can't do video calls on it. The camera only points to me and not the subject. Didn't realize this until recently.
The MacBook delivers top-notch performance! It’s 14 hours of battery life delivers!!! The bright and colorful display all in is a delightfully portable design. It’s truly worthy of the Air name.
I needed a new laptop for school, but I also wanted something powerful enough for some casual gaming here and there and maybe some video editing down the line. This notebook is able to do both of those! And it's light enough that I don't feel weighed down when it's in my school bag. I'm not a fan of the red keyboard, but it's no deal breaker. I upgraded to a Samsung 1tb nvme SSD and added 8gb more ram for 16 total. This thing is officially a little powerhouse! MSI customer support is also amazing, they answer the phone right away and are very knowledgeable about all of their products. There's tons and tons of information about their products on their website and youtube page. Thanks to that, upgrading this notebook was easy for me (who's otherwise not super knowledgeable about computers). Money well spent!

Given the following review, please return the judgement result:
```

### Result

* Pre: 0.3922
* Rec: 1
* F1: 0.5634

|         | Predict=1 | Predict=0 |
| ------- | --------- | --------- |
| Label=1 | 20        | 0         |
| Label=0 | 31        | 69        |





# Generate V2 (LLama)

## Prompt

```python
Given a computer purchase review data from an e-commerce platform, determine whether the review data contains new requirements. New requirements indicate features that the computer does not currently have. Return 1 (contains new requirements) or 0 (does not contain new requirements).
Here are some examples:

Maybe a laptop with multiple foldable screens which can be combined to form a bigger one will be the solution.==>1
I'm utterly frustrated with the laptop's battery life. For a device that boasts high performance, the battery drains faster than I would like. I find myself tethered to a power outlet more often than not. The inclusion of a higher capacity battery or a fast-charge feature would make this laptop truly mobile.==>1
The laptop is a great choice for people who want a portable gaming experience, though there are a few problems. First, the laptop will get loud/hot very fast on newer games unless you lower the quality a lot. Second, the touchpad is sensitive and you'll find yourself accidently clicking things from time to time. Third, the people who shipped my laptop didn't have a sticker on the package warning about the lithium batteries as well as didn't even put any bubble wrap to keep the laptop safe. Finally, the battery life only last a few hours so you'll have to charge regularly. Other than that, the laptop works great and can play any game I throw at it. I would recommend it to anyone who wants an affordable gaming experience==>0
i had to buy this for school but honestly the m1 is a better value, especially refurbished or used. if all you'll be doing is schoolwork, youtube, spotify and mail, this laptop does not feel worth 1000 bucks. i'm too lazy to return it at this point but if i could pick again i would've bought the cheaper m1 macbooks.==>0

Given the following review, please return the judgement result:
```



## Case Study

```python
'Based on the information provided, the review data contains a new requirement. The reviewer mentions that the F10 key is defective right out of the box, indicating that the laptop does not currently have a functioning F10 key. Therefore, the judgment result is 1 (contains new requirements).' 0

==================================

'Based on the review data you provided, the judgment result is 0 (does not contain new requirements). The review does not mention any specific new features or requirements that the computer does not currently have. It simply expresses a positive sentiment towards the laptop, stating that it is "muy bueno" (very good).' 0

==================================

'Based on the review you provided, the judgment result is 1 (contains new requirements).\n\nThe reviewer mentions that they had issues with the shipping process, specifically that the shipping was delayed twice despite their attempts to reschedule and pay for one-day shipping. This indicates that the customer had specific requirements for the shipping process that were not met, such as timely delivery and reliable scheduling. These requirements are not currently met by the laptop itself, but rather by the shipping and delivery process. Therefore, this review contains new requirements.'0

==================================

'Based on the review, it does not contain any new requirements. The user mentions that the laptop is good for their needs, but they do not mention any specific features that the laptop currently lacks. Therefore, the result is 0 (does not contain new requirements).' 0

==================================

"Based on the review, I would return a result of 0 (does not contain new requirements). The review does not mention any specific features that the computer does not currently have, but rather expresses satisfaction with the computer's value for the price." 0

==================================

"Based on the review you provided, the judgment result is 0 (does not contain new requirements). The review does not mention any new features or requirements that the computer does not currently have. It only mentions the laptop's touch and keyboard, which are existing features." 0

==================================

```







# GitHub

https://github.com/Chunhao-Li/novel-need-detection



# novel need

* 新名词
* 新组合





# Others

https://hsuhk.zoom.us/j/7092783994 N9wSYL
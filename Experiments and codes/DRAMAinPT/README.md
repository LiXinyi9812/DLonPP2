`DRAMA`: the reprodduction of DRAMA.  
`DRAMAinPT_feature_extraction_cotrain`: the reproduction of DRAMA, however, here we try to use segmentation images with annotation as an additional input.
Segmentation images are added by concatenating with the original PP2 images. And we also modify the last layer of drama($fc_{input}=10 \times 92 \times 92$),and set batch size=24.   
`DRAMAinPT_feature_extraction_respectively`: The same inputs and architecture as former, but train 6 attributes respectively.  

The codes, results and saved models are in folder with the same name on LIGER.  
The models' performance(the three in blue) are in https://docs.google.com/spreadsheets/d/1Xrfgqes7q3pcsZN1M5sxtDZ9zWNo3Isz/edit?usp=sharing&ouid=116497001749940438866&rtpof=true&sd=true  


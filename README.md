# 3DHARSOM
###### <h4> Author: Zahra Gharaee (zahra.gharaee@liu.se)
  ###### <h3> Overview
 
This repository contains the code implementation of an innovative approach named 3DHARSOM for 3D human action recognition. This approach effectively tackles the action classification challenge by integrating both the first and second orders of dynamics. The input dataset comprises actions captured by the Kinect sensor, specifically the [MSRAction3D Dataset](https://www.microsoft.com/en-us/research/people/zliu/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fzliu%2Factionrecorsrc%2F), which serves as the foundation for conducting experiments. The architecture of 3DHARSOM consists of several layers, including a pre-processing layer, a first-layer Self-Organizing Map (SOM), a superimposition layer, a second-layer SOM, and a third-layer neural network for labeling. Researchers and enthusiasts interested in utilizing the 3DHARSOM architecture or any of its individual components are encouraged to cite the associated article or its [article](https://doi.org/10.1016/j.asoc.2017.06.007)/[arxiv link](https://arxiv.org/abs/2104.06059):

This architecture demonstrates an innovative approach to enhancing 3D human action recognition, offering a comprehensive solution that leverages both spatial and temporal dynamics.
  
      @article {gharaee2017asc} {
          author = {Zahra Gharaee and Peter G{\"{a}}rdenfors and Magnus Johnsson},
            title = {First and second order dynamics in a hierarchical {SOM} system for action recognition},
            booktitle = {Applied Soft Computing},
            year = {2017}
            page = {574--585}
            volume = {59}
            DOI = {10.1016/j.asoc.2017.06.007}
          }
        }
    
  
###### <h3> Run experiment
Run HAR.py scripts and apply the settings required to define the dataset and hyperparameters used for either training or testing the system. A pretrained model is available in directory 'trained_models/MSR_Action3D_2_pretrained', which shows the performance of the system.   
  
###### <h3> Other related works
Vriants of **3DHARSOM** architecture are designed and implemented to address different research problems. These studies are presented in several scientific papers (and demos) mentioned in the following:   
* Application of **3DHARSOM** architecture without using orders of dynamic, [article](https://www.scitepress.org/Papers/2017/61993/61993.pdf): 
  
  
      @inproceedings {gharaee2017icaart} {
              author = {Zahra Gharaee and Peter G{\"{a}}rdenfors and Magnus Johnsson},
              title = {Hierarchical Self-organizing Maps System for Action Classification},
              booktitle = {Proceedings of the 9th International Conference on Agents and Artificial Intelligence, {ICAART}},
                year = {2017}
                page = {583--590}
                publisher = {SciTePress},
                DOI = {10.5220/0006199305830590}
              }
            }
  
   
  
* Application of **3DHARSOM** architecture with manual segmentation while actor performs actions of MSRAction3D dataset infront of a Kinect sensor, [article](https://ieeexplore.ieee.org/abstract/document/7907518):
  
  
      @inproceedings {gharaee2016sitis} {
              author = {Zahra Gharaee and Peter G{\"{a}}rdenfors and Magnus Johnsson},
              title = {Action Recognition Online with Hierarchical Self-Organizing Maps},
              booktitle = {12th International Conference on Signal-Image Technology {\&}
               Internet-Based Systems, {SITIS}},
                year = {2016}
                page = {538--544}
                publisher = {{IEEE} Computer Society},
                DOI = {10.1109/SITIS.2016.91}
              }
            }
 
https://user-images.githubusercontent.com/8222285/124340307-81033780-dbb4-11eb-8da5-70b6339614f2.mov
  

* Application of **3DHARSOM** architecture for action recognition and object detection by running experiments using a Kinect sensor and actions involve objects, [article](https://doi.org/10.1016/j.bica.2017.09.007)/[arxiv link](https://arxiv.org/abs/2104.06070):
  
      @article {gharaee2017bica} {
          author = {Zahra Gharaee and Peter G{\"{a}}rdenfors and Magnus Johnsson},
            title = {Online recognition of actions involving objects},
            booktitle = {Biologically Inspired Cognitive Architectures},
            year = {2017}
            page = {10--19}
            volume = {22}
            DOI = {10.1016/j.bica.2017.09.007}
          }
        }
  
https://user-images.githubusercontent.com/8222285/124340409-582f7200-dbb5-11eb-9644-33b781bb52aa.mov

* Application of **3DHARSOM** architecture for online recognition of unsegmented actions of MSRAction3D dataset, [article](https://doi.org/10.1007/s10339-020-00986-4)/[arxiv link](https://arxiv.org/abs/2104.11637):
  
      @article {gharaee2017cogproc} {
          author = {Zahra Gharaee},
            title = {Online recognition of unsegmented actions with hierarchical SOM architecture},
            booktitle = {Cognitive Processing},
            year = {2021}
            page = {77--91}
            volume = {22}
            DOI = {10.1007/s10339-020-00986-4}
          }
        }

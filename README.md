# DBS2Net: Lightweight JPEG Image Steganalysis Using Dilated Blind-Spot Network


Mingzhi Hu, Hongxia Wang, Lightweight JPEG image steganalysis using dilated blind-spot network, Journal of Visual Communication and Image Representation,
2024, 104182, ISSN 1047-3203, https://doi.org/10.1016/j.jvcir.2024.104182.

(https://www.sciencedirect.com/science/article/pii/S1047320324001378)

**Abstract**: The primary goal of image steganalysis is to detect subtle steganographic signals hidden within digital images. An essential aspect of digital image steganalysis involves the extraction of meaningful steganographic signal features. However, existing JPEG steganalysis networks have not delved deep enough into the research on extracting steganographic noise signals, especially in the preprocessing stage of the model. In this paper, we propose a novel approach called the Dilated Blind-Spot Steganalysis Network (DBS2Net), specifically designed for JPEG steganalysis. DBS2Net incorporates the dilated blind-spot convolutional network into the preprocessing stage to extract noise residuals, thereby enhancing detection accuracy. The blind-spot network estimates central outputs based on adjacent regions to effectively suppress semantic information and uses dilated convolutions to expand the receptive field. Additionally, we design the transition block to further enhance noise signals by mapping noise residuals extracted by the blind-spot network. In feature extraction, depth-wise separable convolutions are used to minimize the parameter count, while dual-view pooling layers compress features, resulting in improved detection performance. Experimental results validate the efficacy of this approach, and comprehensive analyses of JPEG steganographic noise support its success. In particular, DBS2Net outperforms state-of-the-art steganalysis methods, achieving superior detection accuracy with minimal model complexity.
Keywords: Steganalysis; Steganography; Blind-spot network; Deep learning

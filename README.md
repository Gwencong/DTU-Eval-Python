# DTU Evaluation with Python
When conducting point cloud evaluation on the DTU dataset, it is necessary to intall Matlab and use the official Matlab evaluation code, which is very cumbersome.   
Based on https://github.com/jzhangbs/DTUeval-python, Python evaluation code has been implemented


## Usage
1. Install dependency 
```bash
pip install -r requirements.txt
```
2. Prepare Dataset
- Download the STL [Point clouds](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) and [Sample Set](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and unzip them, then copy the `Points/Points/stl/stlxxx_total.ply` file in to the `SampleSet/MVS Data/Points/stl` folder.
- Get prediction results of your mvs algorithm, the naming format of predicted ply file is `{method}{scanid}_l3.ply` such as `mvsnet001_l3.ply`,`casmvsnet001_l3.ply`, or you can name it your own way and modify the reding code [here](http://github.com/Gwencong/utils.py/Line134). 
After the two steps above, your data directory will like bellow:
```
./SampleSet/MVS Data/
|--Points
|   |--stl
|       |--stlxxx_total.ply
|--ObsMask
|   |--ObsMaskxxx_10.mat
|   |--Planexxx.mat

./Predict/
|--mvsnet
|   |--mvsnetxxx_l3.ply
```

3. evaluation
- Use following command to evaluation:
```bash
python dtu_eval.py --method mvsnet --pred_dir "./Preidct/mvsnet/" --gt_dir "./SampleSet/MVS Data"
```
- Note:  
If you encounter a memory shortage issue, it is possible that the version of Python you are installing is 32-bit, which has limitations on memory usage. Therefore, it is recommended to install 64-bit Python

## Compare with MATLAB Evalution result
We compare the evaluation results of matlab and python in r-mvsnet and casmvsnet. The results obtained from this implementation are slightly higher than those from MATLAB code but more fast and memory saving, can be used during experiments. The difference in results between the two is mainly due to random shuffling in the code. Results are shown bellow:

|Method|acc.(mm)|comp.(mm)|overall(mm)|
|------|--------|---------|-----------|
|R-MVSNet(matlab)|0.3835|0.4520|0.4177|
|R-MVSNet(python)|0.3836|0.4581|0.4209|
|CasMVSNet(matlab)|0.3779|0.3636|0.3707|
|CasMVSNet(python)|0.3780|0.3669|0.3739|
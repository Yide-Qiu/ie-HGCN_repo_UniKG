# This is a project used to reproduce ie-HGCN on UniKG.

## To configure the environment, please execute the configuration file that we have prepared:

bash deploy.sh

## Dataset

First, prepare the data：

'./dataset/UniKG_1M.pk'

Alternatively, prepare the data in the '1M' directory：
edge_index_dict.pth
edge_reltype_dict.pth
node_year_dict.pth
num_nodes_dict.pth
x_dict.pth
y_dict.pth

To generate './dataset/UniKG_1M.pk'.

we provide all the data in the UniKG project.


## How to run?

python train.py --data_dir './dataset/UniKG_1M.pk' --model_dir PATH_OF_SAVED_MODEL --n_layers 3 --n_hid 256 --prev_norm --last_norm --conv_name ieHGCN --sample_width 500 --batch_size 256 --cuda 0 --n_batch 32 --n_epoch 200


## Training Record

Please refer to the file 'ie-HGCN-UNIKG-1M.txt' for the replication record.





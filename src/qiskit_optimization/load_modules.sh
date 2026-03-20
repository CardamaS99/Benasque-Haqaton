conda deactivate
module load qmio/hpc  gcc/12.3.0  matplotlib jupyter-bundle qmio-run  matplotlib
module load qiskit  qmio-tools


if [[ "$1" == "--ins" ]]; then
    cd /mnt/netapp1/Store_CESGA/home/cesga/mlosada/api/api-simulator
    cmake --install build/
    cd -
fi
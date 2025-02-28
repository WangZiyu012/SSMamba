import re
import matplotlib.pyplot as plt

x_size = 18
# File paths for the uploaded logs
base_add = ""
#file_paths = [base_add + "PM2AE.log", base_add + "PM2AE-SA3DF.log", base_add + "PMAE.log", base_add + "PMAE-SA3DF.log"]

###file_paths = [base_add + "method_k_top_eigenvectors__reverse_True__knn_graph_10__k_small_eigenvectors_4_PRETRAIN_2.log", base_add + "metod_k_top_eigenvectors_seperate_learnable_tokens__reverse_True__knn_graph_20__k_small_eigenvectors_4__lr_1e3_Pretrain.log"]
file_paths = [base_add + "method_k_top_eigenvectors__reverse_True__knn_graph_10__k_top_eigenvectors_4_lr_1e3__FINETUNE_WITH_TOP_PRETRAIN.log", base_add + "metod_k_top_eigenvectors__reverse_True__knn_graph_20__k_small_eigenvectors_4__lr_5e4_FINETUNE_OUR_METHOD_BEST_MODEL_METHOD_SEPERATED.log"]
#file_paths = [base_add + "PMAE.log", base_add + "PMAE-SA3DF.log"]     
##file_paths = [base_add + "PMAE.log", base_add + "PMAE-SA3DP.log", base_add + "PMAE-SA3DF.log"] 

# Function to extract accuracy values from a given file
def extract_accuracies_v2(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            # Searching for the accuracy patterns in each line
            match = re.search(r'Linear Accuracy : (\d+\.\d+)|"val_acc": (\d+\.\d+)|acc = (\d+\.\d+)|"val_svm_acc": (\d+\.\d+)', line)
            if match:
                # Extract the matched accuracy value 
                matched_groups = match.groups()
                accuracy = next((float(m) for m in matched_groups if m is not None), None)
                if accuracy:
                    #accuracies.append(accuracy*100)
                    accuracies.append(accuracy)
    return accuracies

# Extract accuracies from each log file
accuracies_all_files = [extract_accuracies_v2(file_path) for file_path in file_paths]

# Plotting the accuracies
plt.figure(figsize=(10, 6))
#colors = ['b', 'g', 'r', 'c']  # Different colors for each file
#labels = ['PM2AE', 'PM2AE-SA3DF', 'PMAE', 'PMAE-SA3DF']  # Labels for each file

#colors = ['b', 'g']  # Different colors for each file
#labels = ['PM2AE', 'PM2AE-SA3DF']  # Labels for each file


########################################################################
# colors = ['r', 'b', 'g']  # Different colors for each file
# labels = ['PointMAE', 'PointMAE+GM3D*', 'PointMAE+GM3D']  # Labels for each file
# #labels = ['PointM2AE', 'PointM2AE+SA3DF']  # Labels for each file

# for accuracies, color, label in zip(accuracies_all_files, colors, labels):
#     epochs = range(1, len(accuracies[:40]) + 1)
#     plt.plot(epochs, accuracies[:40], color=color, label=label, linewidth=3)

# plt.xlabel('Epochs', fontsize=16)
# plt.ylabel('SVM Accuracy', fontsize=16)
# plt.title('Accuracy vs Epochs for Different Models', fontsize=16)
# plt.legend(fontsize=16)
# plt.grid(False)
# plt.savefig(base_add + "NEW_ECCV2024_1.png")
##########################################################################

##colors = ['#F0746E', '#7CCBA2', '#045275']  # Different colors for each file
##labels = ['Point-MAE', 'Point-MAE+GM3D*', 'Point-MAE+GM3D']  # Labels for each file

#colors = ['#F0746E', '#045275']  # Different colors for each file
colors = ['tab:blue', 'xkcd:sky blue']  # Different colors for each file
labels = ['Ours without TAR', 'Ours with TAR']  # Labels for each file

for accuracies, color, label in zip(accuracies_all_files, colors, labels):
    epochs = range(1, len(accuracies[:300]) + 1)
    plt.plot(epochs, accuracies[:300], color=color, label=label, linewidth=3)

        # Find the maximum accuracy and its corresponding epoch
    max_accuracy = max(accuracies)
    max_epoch = accuracies.index(max_accuracy) + 1  # +1 because epochs start at 1

    # Plot a dot at the maximum point
    #plt.plot(max_epoch, max_accuracy, 'o', color=color)

    """# Optionally, add an annotation with an arrow pointing to the dot
    plt.annotate(f'Max: {max_accuracy:.2f}',z
                 xy=(max_epoch, max_accuracy),
                 xytext=(max_epoch + 5, max_accuracy),
                 arrowprops=dict(facecolor=color, shrink=0.05))"""
    
    # Plot a short dashed line at the maximum accuracy point
    plt.plot([max_epoch - 20, max_epoch + 20], [max_accuracy, max_accuracy], '--', color=color, lw=2)

    # Add text for the maximum accuracy, adjusted to be shown with a percentage
    plt.text(max_epoch + 5, max_accuracy, f'{max_accuracy:.2f}', color="black", va='bottom', ha='center', fontsize=18)

# Remove the top and right spines
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set the limits for your axes if not already set
ax.set_xlim(left=0, right=ax.get_xlim()[1])
ax.set_ylim(bottom=0, top=ax.get_ylim()[1])


# Hide the ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add arrow to end of the x-axis
ax.annotate('', xy=(0.99, 0), xycoords='axes fraction', xytext=(10, 0), textcoords='offset points',
            arrowprops=dict(arrowstyle="<-", color='black'))

# Add arrow to end of the y-axis
ax.annotate('', xy=(0, 0.98), xycoords='axes fraction', xytext=(0, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="<-", color='black'))

# ax.set_xlim([xmin, xmax])
##ax.set_ylim([0.55, .95])
#ax.set_ylim([80, 92])
ax.set_ylim([55, 95])
#ax.set_xlim([1, 250])
ax.set_xlim([1, 300])

#plt.xlabel('Epochs', fontsize=14, fontstyle='italic', labelpad=10)  # labelpad for distance from axis
plt.xlabel('Epochs', fontsize=x_size, labelpad=10)
#plt.ylabel('SVM Accuracy', fontsize=14, fontstyle='italic', labelpad=10)  # labelpad for distance from axis
#plt.ylabel('SVM Accuracy', fontsize=18, labelpad=10)
plt.ylabel('Overall Accuracy', fontsize=x_size, labelpad=10)


# Adjust the label positions
#ax.yaxis.set_label_coords(-0.1, 1.02)  
#ax.xaxis.set_label_coords(1.05, -0.02)

ax.tick_params(axis='x', labelsize=x_size) 
ax.tick_params(axis='y', labelsize=x_size)

#plt.title('Accuracy vs Epochs for Different Models', fontsize=x_size)   
plt.legend(fontsize=x_size, loc='lower right') #loc='lower right'
plt.grid(False)
plt.savefig(base_add + "ours_w_wo_tar_downstream_2.pdf", bbox_inches='tight')  # bbox_inches='tight' for fitting the label in the saved figure


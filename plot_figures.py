import torch
import matplotlib.pyplot as plt

# Assuming you have PyTorch tensors
#k_smallest_eigenvectors = torch.tensor([2, 3, 4, 5, 6, 7])
#overall_accuracy = torch.tensor([87.6, 88.46, 90.18, 87.77, 87.95, 88.46])
x_size = 18
k_smallest_eigenvectors = torch.tensor([2, 3, 4, 5])
overall_accuracy = torch.tensor([87.6, 88.46, 90.18, 88.64])

# Convert tensors to numpy arrays for plotting
x = k_smallest_eigenvectors.numpy() 
y = overall_accuracy.numpy()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='*', linestyle='--', color='xkcd:sky blue', label='Ours', linewidth=4, markersize=10)
plt.axhline(y=86.75, color='tab:orange', linestyle='--', label='Point-MAE', linewidth=4)  # Add horizontal line
plt.axhline(y=88.30, color='tab:green', linestyle='--', label='Point-MAMBA', linewidth=4)  # Add horizontal line
plt.axhline(y=87.95, color='tab:brown', linestyle='--', label='Random-MAMBA', linewidth=4)  # Add horizontal line
plt.xlabel('Number of Smallest Eigenvectors', fontsize=x_size)
plt.ylabel('Overall Accuracy', fontsize=x_size)
#plt.title('Overall Accuracy vs Number of Smallest Eigenvectors', fontsize=x_size)
plt.legend(fontsize=x_size)


# Customize the plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Hide the ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add arrow to end of the x-axis
ax.annotate('', xy=(0.99, 0), xycoords='axes fraction', xytext=(10, 0), textcoords='offset points',
            arrowprops=dict(arrowstyle="<-", color='black'))

# Add arrow to end of the y-axis
ax.annotate('', xy=(0, 0.98), xycoords='axes fraction', xytext=(0, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="<-", color='black'))


# Removing the background
ax.patch.set_visible(False)
ax.set_xticks([2, 3, 4, 5])

ax.tick_params(axis='x', labelsize=x_size) 
ax.tick_params(axis='y', labelsize=x_size)

# Save the plot as a PDF file
plt.savefig('overall_accuracy_vs_eigenvectors.pdf', bbox_inches='tight', transparent=True)



############### knn
# Assuming you have PyTorch tensors
#k_smallest_eigenvectors = torch.tensor([5, 10, 15, 20])
#overall_accuracy = torch.tensor([88.12, 88.64, 88.05, 88.64])
k_smallest_eigenvectors = torch.tensor([10, 15, 20, 25, 30])
overall_accuracy = torch.tensor([87.26, 88.12, 90.18, 88.81, 88.29])

# Convert tensors to numpy arrays for plotting
x = k_smallest_eigenvectors.numpy()
y = overall_accuracy.numpy()


plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='*', linestyle='--', color='xkcd:sky blue', label='Ours', linewidth=4, markersize=10)
#plt.axhline(y=86.23, color='tab:brown', linestyle='--', label='Random-MAMBA')  # Add horizontal line
#plt.axhline(y=86.92, color='tab:orange', linestyle='--', label='Point-MAE')  # Add horizontal line
#plt.axhline(y=87.78, color='tab:green', linestyle='--', label='Point-MAMBA')  # Add horizontal line
plt.xlabel('K Nearest Neighbors', fontsize=x_size)
plt.ylabel('Overall Accuracy', fontsize=x_size)
#plt.title('Overall Accuracy vs K Nearest Neighbors', fontsize=x_size)
#plt.legend(fontsize=12)


# Customize the plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Add arrow to end of the x-axis
ax.annotate('', xy=(0.99, 0), xycoords='axes fraction', xytext=(10, 0), textcoords='offset points',
            arrowprops=dict(arrowstyle="<-", color='black'))

# Add arrow to end of the y-axis
ax.annotate('', xy=(0, 0.98), xycoords='axes fraction', xytext=(0, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="<-", color='black'))


# Removing the background
ax.patch.set_visible(False)
ax.set_xticks([10, 15, 20, 25, 30])

ax.tick_params(axis='x', labelsize=x_size) 
ax.tick_params(axis='y', labelsize=x_size)


# Save the plot as a PDF file
plt.savefig('overall_accuracy_vs_knn.pdf', bbox_inches='tight', transparent=True)


###################
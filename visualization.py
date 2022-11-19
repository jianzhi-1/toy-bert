# Loss graph
plt.figure(figsize=(15,10))
plt.plot(np.arange(len(loss_graph)), loss_graph)
plt.xlabel('Batches') 
plt.ylabel('Loss')
plt.legend()
plt.title('Plot of Loss against Batches')
plt.show()

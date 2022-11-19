# Loss graph
plt.figure(figsize=(15,10))
plt.plot(np.arange(len(loss_graph)), loss_graph)
plt.xlabel('Batches') 
plt.ylabel('Loss')
plt.legend()
plt.title('Plot of Loss against Batches')
plt.show()


# TSNE on Embedding Layer
from sklearn import datasets, manifold
embedding_numpy = np.array(state["params"]["embed"]["embeddings"])
tsne = manifold.TSNE(n_components=2, random_state=42)
embedding_numpy = tsne.fit_transform(embedding_numpy)
plt.figure(figsize=(15,10))
plt.scatter(embedding_numpy[:,0], embedding_numpy[:,1], cmap='viridis')
for i in range(len(embedding_numpy)):
  plt.annotate(inv_map[i], (embedding_numpy[i][0], embedding_numpy[i][1]))
plt.colorbar()
plt.title("Scatter Plot of t-SNE (annotated)")
plt.show()

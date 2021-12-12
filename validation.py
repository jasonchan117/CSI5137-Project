class Validation:
	p_order = ["NFR", "F"]
	c_order = ["F", "A", "FT", "L", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US"]
	def __init__(self, p_labels, c_labels, out_p_labels, out_c_labels):
		'''
		@param: p_labels: the parent labels, array of bitmaps
		@param: c_labels: the child labels, array of bitmaps
		@param: out_p_labels: the output of classification model (parent labels), array of one-hot codes
		@param: out_c_labels: the output of classification model (child labels), array of one-hot codes
		'''
		self.p_labels = p_labels
		self.c_labels = c_labels
		### Convert One-Hot Code to BitMap
		self.out_p_labels = list(list(1 if nb == max(ls) else 0 for nb in ls) for ls in out_p_labels)
		### Convert One-Hot Code to BitMap
		self.out_c_labels = list(list(1 if nb == max(ls) else 0 for nb in ls) for ls in out_c_labels)

		### Get Vadilation Data
		self.p_precision, self.c_precision = self.get_precision()
		self.p_recall, self.c_recall = self.get_recall()
		self.p_f, self.c_f = self.get_f_score()
	
	def get_precision(self):
		'''
		Precision = TP/(TP+FP)
		@return p_precision: array of F's precision and NFR's precision (length: 2)
		@return c_precision: array of precisions for all child labels   (length: 12)
		'''
		p_precision = []
		for each_p_label in range(0, len(self.p_labels[0])):
			TP = 0
			TPnFP = 0
			for index in range(0, len(self.p_labels)):
				if self.out_p_labels[index][each_p_label] == 1:
					TPnFP += 1
					if self.p_labels[index][each_p_label] == 1:
						TP += 1
			p_precision.append(TP/TPnFP)

		c_precision = []
		for each_c_label in range(0, len(self.c_labels[0])):
			TP = 0
			TPnFP = 0
			for index in range(0, len(self.c_labels)):
				if self.out_c_labels[index][each_c_label] == 1:
					TPnFP += 1
					if self.c_labels[index][each_c_label] == 1:
						TP += 1
			c_precision.append(TP/TPnFP)
		return p_precision, c_precision

	def get_recall(self):
		'''
		Recall = TP/(TP+FN)
		@return p_recall: array of F's recall and NFR's recall (length: 2)
		@return c_recall: array of recalls for all child labels   (length: 12)
		'''
		p_recall = []
		for each_p_label in range(0, len(self.p_labels[0])):
			TP = 0
			TPnFN = 0
			for index in range(0, len(self.p_labels)):
				if self.p_labels[index][each_p_label] == 1:
					TPnFN += 1
					if self.out_p_labels[index][each_p_label] == 1:
						TP += 1
			p_recall.append(TP/TPnFN)

		c_recall = []
		for each_c_label in range(0, len(self.c_labels[0])):
			TP = 0
			TPnFN = 0
			for index in range(0, len(self.c_labels)):
				if self.c_labels[index][each_c_label] == 1:
					TPnFN += 1
					if self.out_c_labels[index][each_c_label] == 1:
						TP += 1
			c_recall.append(TP/TPnFN)
		return p_recall, c_recall

	def get_f_score(self):
		'''
		F = 2*precision*recall / (precision + recall)
		@return p_f: array of F's f score and NFR's f score (length: 2)
		@return c_f: array of f scores for all child labels   (length: 12)
		'''
		p_f = []
		for each_p_label in range(0, len(self.p_labels[0])):
			p = self.p_precision[each_p_label]
			r = self.p_recall[each_p_label]
			p_f.append(2*p*r/(p+r))
		c_f = []
		for each_c_label in range(0, len(self.c_labels[0])):
			p = self.c_precision[each_c_label]
			r = self.c_recall[each_c_label]
			c_f.append(2*p*r/(p+r))
		return p_f,c_f

	def output(self):
		print("Precisions on Parent Labels:")
		print(" ".join(self.p_order[x] + ": "+str(self.p_precision[x]) for x in range(0, len(self.p_order))))
		print("Recalls on Parent Labels:")
		print(" ".join(self.p_order[x] + ": "+str(self.p_recall[x]) for x in range(0, len(self.p_order))))
		print("F Scores on Parent Labels:")
		print(" ".join(self.p_order[x] + ": "+str(self.p_f[x]) for x in range(0, len(self.p_order))))

		print("Precisions on Child Labels:")
		print(" ".join(self.c_order[x] + ": "+str(self.c_precision[x]) for x in range(0, len(self.c_order))))
		print("Recalls on Child Labels:")
		print(" ".join(self.c_order[x] + ": "+str(self.c_recall[x]) for x in range(0, len(self.c_order))))
		print("F Scores on Child Labels:")
		print(" ".join(self.c_order[x] + ": "+str(self.c_f[x]) for x in range(0, len(self.c_order))))


'''
For Testing Purpose:

def main():

if __name__ == "__main__":
    main()
'''
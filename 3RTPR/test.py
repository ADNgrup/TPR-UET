class Node:
	def __init__ (self, value, nextadd=None):
		self.value = value
		self.next = nextadd




def print_link_list(head:Node):
	i = 0
	cur_pointer = head
	while True:
		print(f"\t\t----current element is {i}-th position in  list with value {cur_pointer.value}---")
		i += 1
		if cur_pointer.next is None: 
			print("\t\t------end of linklist---")
			break
		else:
			cur_pointer = cur_pointer.next




if __name__ == "__main__":
	n = 5
	inputs = [3, 4, 5,1, 2]
	head = None

	for i in range(5):
		tmp_node = Node(inputs[i], nextadd=None)
		if head is None: 
			head=tmp_node  #assign address of the first node to head_pointer
		else:
			cur_pointer = pre_pointer = head
			flag_insert = False
			while True:
				
				if cur_pointer.value > tmp_node.value:  #assign tmp node before current_node				
					tmp_node.next    = cur_pointer
					if head == cur_pointer:
						print("\t\t==> Change header")
						head = tmp_node
					else: pre_pointer.next = tmp_node
					flag_insert = True
					print(f"----insert {inputs[i]} into linklist in case 1")
					print_link_list(head)
				else:
					if cur_pointer.next is None: #all values in Linklist <= tmp_node ---> append tmp_node into linklist
						cur_pointer.next = tmp_node
						flag_insert = True
						print(f"----insert {inputs[i]} into linklist in case 2")
						print_link_list(head)
					else:
						pre_pointer = cur_pointer
						cur_pointer = cur_pointer.next
						
				

				#check condition for loop:
				if flag_insert: break
	


					

			

				

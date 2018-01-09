import matplotlib.pyplot as plt
import pickle

rate_list_lin, rate_list_auto, interval_list = pickle.load(open("save2.p",
                                                                "rb"))

data_list = zip(interval_list, rate_list_lin, rate_list_auto)
data_list.sort()
interval_list, rate_list_lin, rate_list_auto = zip(*data_list)

#plt.loglog(interval_list, rate_list_lin, label="linear request",  basex=10)
#plt.loglog(interval_list, rate_list_auto, label="auto request",  basex=10)
plt.plot(interval_list, rate_list_lin, label="linear request")
plt.plot(interval_list, rate_list_auto, label="auto request")

plt.legend()
plt.xlabel('Manual Annotation Effort')
plt.ylabel('Success Rate (iou > 0.66)')
plt.title('Effort vs Accuracy')
plt.show()
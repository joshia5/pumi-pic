/************** Host Communication functions **************/
template <typename Space> using IsHost =
  typename std::enable_if<Kokkos::SpaceAccessibility<typename Space::memory_space,
                                                     Kokkos::HostSpace>::accessible, int>::type;
//Send
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Send(ViewT view, int offset, int size,
                                       int dest, int tag, MPI_Comm comm) {
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Send(view.data() + offset, size*size_per_entry,
                  MpiType<BT<ViewType<ViewT> > >::mpitype(), dest, tag, comm);
}
//Recv
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Recv(ViewT view, int offset, int size,
                                       int sender, int tag, MPI_Comm comm) {
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Recv(view.data() + offset, size*size_per_entry, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                  sender, tag, comm, MPI_STATUS_IGNORE);
}
//Isend
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Isend(ViewT view, int offset, int size,
                                        int dest, int tag, MPI_Comm comm, MPI_Request* req) {
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Isend(view.data() + offset, size*size_per_entry, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                   dest, tag, comm, req);
}
//Irecv
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Irecv(ViewT view, int offset, int size,
                                        int sender, int tag, MPI_Comm comm, MPI_Request* req) {
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Irecv(view.data() + offset, size*size_per_entry,
                   MpiType<BT<ViewType<ViewT> > >::mpitype(),
                   sender, tag, comm, req);
}

//Wait
template <typename Space>
IsHost<Space> PS_Comm_Wait(MPI_Request* req, MPI_Status* stat) {
  return MPI_Wait(req, stat);
}

//Waitall
template <typename Space>
IsHost<Space> PS_Comm_Waitall(int num_reqs, MPI_Request* reqs, MPI_Status* stats) {
  return MPI_Waitall(num_reqs, reqs, stats);
}
//Alltoall
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Alltoall(ViewT send, int send_size,
                                           ViewT recv, int recv_size,
                                           MPI_Comm comm) {
  return MPI_Alltoall(send.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      recv.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(), comm);
}

template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Ialltoall(ViewT send, int send_size,
                                            ViewT recv, int recv_size,
                                            MPI_Comm comm, MPI_Request* request) {
  return MPI_Ialltoall(send.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      recv.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      comm, request);

}

//reduce
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Reduce(ViewT send_view, ViewT recv_view, int count,
                                         MPI_Op op, int root, MPI_Comm comm) {
  return MPI_Reduce(send_view.data(), recv_view.data(), count,
                    MpiType<BT<ViewType<ViewT> > >::mpitype(),
                    op, root, comm);

}

//allreduce
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Allreduce(ViewT send_view, ViewT recv_view, int count,
                                            MPI_Op op, MPI_Comm comm) {
  return MPI_Allreduce(send_view.data(), recv_view.data(), count,
                       MpiType<BT<ViewType<ViewT> > >::mpitype(), op, comm);
}

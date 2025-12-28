use num_traits::Zero;
use std::cell::{Cell, UnsafeCell};
use std::fmt::Debug;
use std::mem::MaybeUninit;

/**
 * WengertLists are indexed with [`usize`].
 */
type Index<T> = *const Operation<T>;

#[repr(transparent)]
struct Slot<T>(UnsafeCell<MaybeUninit<Operation<T>>>);

/**
 * A list of operations performed in a forward pass of a dynamic computational graph,
 * used for Reverse Mode Automatic Differentiation.
 *
 * This is dynamic, as in, you build the [Wengert list](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)
 * at runtime by performing operations like addition and multiplication on
 * [Records](Record) that were created with that Wengert list.
 *
 * When you perform a backward pass to obtain the gradients you travel back up the
 * computational graph using the stored intermediate values from this list to compute
 * all the gradients of the inputs and every intermediate step with respect to an output.
 *
 * Although sophisticated implementations can make the Wengert list only log(N) in length
 * by storing only some of the intermediate steps of N computational steps, this implementation
 * is not as sophisticated, and will store all of them.
 *
 * # Panics
 *
 * Every operation and nearly every method a Record has involves manipulating the
 * record's history on its referenced WengertList. This WengertList itself maintains
 * a [RefCell] which tracks borrows at runtime rather than compile time. This is neccessary to
 * maintain the illusion that Records are just ordinary numbers, and the side effects of doing
 * arithmetic with Records are limited to their referenced WengertList. Hence, the Rust
 * compiler correctly infers that it is not safe to share references to WengertLists between
 * threads, nor transfer Records across threads. If you called a method on two Records that both
 * mutably borrowed from the same WengertList at once, which could be trivially done with
 * multiple threads, then the code would panic. Easy ML shouldn't allow you to do this
 * in safe Rust because each mutable borrow of the WengertList is dropped at the end of each
 * Record method call, and you can't call two methods simulatenously without threading.
 */
pub struct WengertList<T: Copy> {
    operations: Box<[Slot<T>]>,
    last_operation: Cell<usize>,
}

/**
 * A binary operation to record on a WengertList. For unary operations the
 * right derivative is set to 0, and for nullary operations both derivatives
 * are set to 0.
 *
 * Each operation acts like a node in an upside down binary tree, with two parents that
 * each node was computed from. The main difference is that the numerical
 * index of those parents in the WengertList is stored, rather than any pointers.
 */
#[derive(Debug, Copy, Clone)]
struct Operation<T> {
    left_parent: Index<T>,
    right_parent: Index<T>,
    left_derivative: T,
    right_derivative: T,
}

struct BorrowedWengertList<'a, T> {
    slot: &'a mut MaybeUninit<Operation<T>>,
}

/**
 * A wrapper around a real number which records it going through the computational
 * graph. This is used to perform Reverse Mode Automatic Differentiation.
 *
 * # Panics
 *
 * Every operation and nearly every method a Record has involves manipulating the
 * record's history on its referenced [WengertList]. This WengertList itself maintains
 * a [RefCell] which tracks borrows at runtime rather than compile time. This is neccessary to
 * maintain the illusion that Records are just ordinary numbers, and the side effects of doing
 * arithmetic with Records are limited to their referenced WengertList. Hence, the Rust
 * compiler infers that it is not safe to share references to WengertLists between threads,
 * nor transfer Records across threads. If you called a method on two Records that both
 * mutably borrowed from the same WengertList at once, which could be trivially done with
 * multiple threads, then the code would panic. Easy ML shouldn't allow you to do this
 * in safe Rust because each mutable borrow of the WengertList is dropped at the end of each
 * Record method call, and you can't call two methods simulatenously without threading.
 *
 * # Acknowledgments
 *
 * A [tutorial by Rufflewind](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
 * and the associated [MIT licensed](http://opensource.org/licenses/MIT)
 * [source code](https://github.com/Rufflewind/revad/blob/master/src/tape.rs) were invaluable
 * in providing understanding on how to implement Reverse Mode Automatic Differentiation.
 */
#[repr(C)]
pub struct Record<'a, T: Copy> {
    // A record consists of a number used in the forward pass, as
    // well as a WengertList of operations performed on the numbers
    // and each record needs to know which point in the history they
    // are for.
    /**
     * The real number
     */
    pub number: T,
    pub(super) history: Option<&'a WengertList<T>>,
    /**
     * The index of this number in its [WengertList]. The first entry will be 0,
     * the next 1 and so on.
     *
     * In normal use cases you should not need to read this field,
     * you can index [Derivatives] directly with Records.
     */
    index: Index<T>,
}

#[repr(C)]
pub struct FrozenRecord<T: 'static> {
    pub number: T,
    /// Do not touch this field at all.
    /// If value was transmuted from ordinary `[Record<T>]` here will be potentially dangling pointer.
    _history: Option<&'static std::convert::Infallible>,
    pub index: usize,
}

pub(super) mod impls_list {
    use std::cell::Cell;
    use std::fmt::{Debug, Formatter};
    use num_traits::Zero;
    use crate::record::{BorrowedWengertList, Index, Operation, Record, WengertList};

    impl<T: Copy> Debug for WengertList<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("WengertList")
                .field("capacity", &self.operations.len())
                .finish_non_exhaustive()
        }
    }

    impl<T: Copy> WengertList<T> {
        pub fn new() -> Self {
            Self::with_capacity(100_000)
        }

        /**
         * Creates a new empty WengertList from which Records can be constructed.
         */
        pub fn with_capacity(capacity: usize) -> WengertList<T> {
            WengertList {
                operations: unsafe {
                    std::mem::transmute(Box::<[Operation<T>]>::new_uninit_slice(capacity))
                },
                last_operation: Cell::new(0),
            }
        }

        pub fn leak(capacity: usize) -> &'static WengertList<T> {
            let list = Box::new(WengertList::with_capacity(capacity));
            Box::leak(list)
        }
    }

    impl<T: Copy> WengertList<T> {
        /**
         * Clears a WengertList to make it empty again. After clearing a WengertList
         * you must reset all the Records still using that list. Then you can perform
         * another computation and get new gradients.
         */
        pub fn clear(&self) {
            self.last_operation.set(0);
        }
    }

    impl<T: Copy> WengertList<T> {
        /**
         * Creates a record backed by this WengertList.
         *
         * You can alternatively use the [record constructor on the Record type](Record::variable()).
         */
        pub fn variable(&self, x: T) -> Record<'_, T>
        where
            T: Zero,
        {
            Record {
                number: x,
                history: Some(self),
                index: self.append_nullary(),
            }
        }

        /**
         * Adds a value to the list which does not have any parent values.
         */
        pub(super) fn append_nullary(&self) -> Index<T>
        where
            T: Zero,
        {
            self.borrow(|it| it.append_nullary())
        }

        /**
         * Adds a value to the list which has one parent.
         *
         * For an output w_N which depends on one parent w_N-1
         * the derivative cached here is δw_N / δw_N-1
         *
         * For example, if z = sin(x), then δz/δx = cos(x)
         */
        pub(super) fn append_unary(&self, parent: Index<T>, derivative: T) -> Index<T>
        where
            T: Zero,
        {
            self.borrow(|it| it.append_unary(parent, derivative))
        }

        /**
         * Adds a value to the list which has two parents.
         *
         * For an output w_N which depends on two parents w_N-1
         * and w_N-2 the derivatives cached here are δw_N / δw_N-1
         * and δw_N / δw_N-2.
         *
         * For example, if z = y + x, then δz/δy = 1 and δz/δx = 1
         * For example, if z = y * x, then δz/δy = x and δz/δ/x = y
         */
        pub(super) fn append_binary(
            &self,
            left_parent: Index<T>,
            left_derivative: T,
            right_parent: Index<T>,
            right_derivative: T,
        ) -> Index<T> {
            self.borrow(|it| {
                it.append_binary(left_parent, left_derivative, right_parent, right_derivative)
            })
        }

        /**
         * Borrows the WengertList mutably for batch operations. It is *very* important to
         * hold onto the borrow only for as long as needed then drop it immediately. To avoid panics
         * Easy ML needs to ensure 100% of method calls on the public API do not maintain a borrow
         * after they finish executing. This was previously enforced by not having any batch
         * append APIs, but they're needed for RecordContainer. Calling borrow again while still
         * holding the first would trigger a panic, as would holding onto the borrow after the public
         * API method is finished
         */
        #[inline(always)]
        fn borrow<F, R>(&self, op: F) -> R
        where
            F: FnOnce(&mut BorrowedWengertList<T>) -> R,
        {
            let index = self.last_operation.get();
            let slot = &self.operations[index];
            self.last_operation.update(|x| x + 1);

            slot.with_mut(|it| op(&mut BorrowedWengertList::new(it)))
        }
    }
}

pub(super) mod impls_record {
    use std::fmt::{Debug, Formatter};
    use std::ops::AddAssign;
    use num_traits::{One, Zero};
    use crate::{Derivatives, Indexed};
    use crate::record::{FrozenRecord, Record, WengertList};
    use crate::record_operations::same_list;

    impl<T: Debug + Copy> Debug for Record<'_, T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            self.number.fmt(f)
        }
    }

    impl<T: Debug> Debug for FrozenRecord<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            self.number.fmt(f)
        }
    }

    /**
     * The main set of methods for using Record types for Reverse Differentiation.
     *
     * The general steps are
     * 1. create a `WengertList`
     * 2. create variables from this list
     * 3. do operations on the variables
     * 4. from the output you want to compute derivatives for call `.derivatives()`
     * 5. index the `Derivatives` object with the index variables to get the derivatives
     * with respect to each input
     * 6. if you want to make another pass call `clear()` on the `WengertList`
     * and then call `reset()` on all of the variables to forget the gradients already
     * computed (the order of `clear` then `reset` is very important!).
     *
     * Constants can be used to save memory if you have numbers that
     * you do not need to compute the gradients with respect to.
     */
    impl<'a, T: Copy> Record<'a, T> {
        /**
         * Creates an untracked Record which has no backing WengertList.
         *
         * This is provided for using constants along with Records in operations.
         *
         * For example with y = x + 4 the computation graph could be conceived as
         * a y node with parent nodes of x and 4 combined with the operation +.
         * However there is no need to record the derivatives of a constant, so
         * instead the computation graph can be conceived as a y node with a single
         * parent node of x and the unary operation of +4.
         *
         * This is also used for the type level constructors required by Numeric
         * which are also considered constants.
         */
        pub fn constant(c: T) -> Record<'a, T> {
            Record {
                number: c,
                history: None,
                index: std::ptr::null(),
            }
        }

        /**
         * Creates a record backed by the provided WengertList.
         *
         * The record cannot live longer than the WengertList, hence
         * the following example does not compile
         *
         * ```compile_fail
         * use easy_ml::differentiation::Record;
         * use easy_ml::differentiation::WengertList;
         * let record = {
         *     let list = WengertList::new();
         *     Record::variable(1.0, &list)
         * }; // list no longer in scope
         * ```
         *
         * You can alternatively use the [record constructor on the WengertList type](WengertList::variable()).
         */
        pub fn variable(x: T, history: &'a WengertList<T>) -> Record<'a, T>
        where
            T: Zero,
        {
            Record {
                number: x,
                history: Some(history),
                index: history.append_nullary(),
            }
        }

        /**
         * Resets this Record to place it back on its WengertList, for use
         * in performing another derivation after clearing the WengertList.
         */
        pub fn reset(&mut self)
        where
            T: Zero,
        {
            match self.history {
                None => (), // noop
                Some(history) => self.index = history.append_nullary(),
            };
        }

        /**
         * A convenience helper function which takes a Record by value and
         * calls [reset](Record::reset()) on it.
         */
        pub fn do_reset(mut x: Record<T>) -> Record<T>
        where
            T: Zero,
        {
            x.reset();
            x
        }

        /**
         * Gets the WengertList this Record is backed by if a variable, and [None] if a constant.
         */
        pub fn history(&self) -> Option<&'a WengertList<T>> {
            self.history
        }

        pub fn freeze(self) -> FrozenRecord<T> {
            FrozenRecord {
                number: self.number,
                _history: None,
                index: self.index.addr(),
            }
        }
    }

    impl<'a, T: Copy> Record<'a, T> {
        /**
         * Performs a backward pass up this record's WengertList from this
         * record as the output, computing all the derivatives for the inputs
         * involving this output.
         *
         * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is y,
         * then this computes all the derivatives δy/δx<sub>i</sub> for i = 1 to N.
         *
         * # Panics
         *
         * Panics if the Record has no backing WengertList, ie it was created as a
         * constant.
         */
        #[track_caller]
        pub fn derivatives(&self) -> Derivatives<T>
        where
            T: Clone + Zero + One + AddAssign,
        {
            match self.try_derivatives() {
                None => panic!("Record has no WengertList to find derivatives from"),
                Some(d) => d,
            }
        }

        /**
         * Performs a backward pass up this record's WengertList from this
         * record as the output, computing all the derivatives for the inputs
         * involving this output.
         *
         * If this record has no WengertList, ie it's a constant, None is returned instead.
         *
         * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is y,
         * then this computes all the derivatives δy/δx<sub>i</sub> for i = 1 to N.
         */
        pub fn try_derivatives(&self) -> Option<Derivatives<T>>
        where
            T: Clone + Zero + One + AddAssign,
        {
            let history = self.history?;
            let operations = &history.operations;
            let len = history.last_operation.get();

            let mut adjoints = vec![T::zero(); len];

            // δy/δy = 1
            adjoints[self.index()] = T::one();

            let start = operations[0].as_ptr();
            for i in (0..len).rev() {
                operations[i].with_mut(|it| {
                    let operation = unsafe { it.assume_init_ref() };
                    let derivative =
                        adjoints[unsafe { it.as_ptr().offset_from_unsigned(start) }].clone();
                    if !operation.left_parent.is_null() {
                        let left_parent_index =
                            unsafe { operation.left_parent.offset_from_unsigned(start) };
                        adjoints[left_parent_index] +=
                            derivative.clone() * operation.left_derivative.clone();
                    }
                    if !operation.right_parent.is_null() {
                        let right_parent_index =
                            unsafe { operation.right_parent.offset_from_unsigned(start) };
                        adjoints[right_parent_index] +=
                            derivative * operation.right_derivative.clone();
                    }
                });
            }

            Some(Derivatives { adjoints })
        }
    }

    impl<T: Copy> Indexed for Record<'_, T> {
        #[inline(always)]
        fn index(&self) -> usize {
            if let Some(history) = self.history {
                let start = history.operations[0].as_ptr();
                return unsafe { self.index.offset_from_unsigned(start) };
            }
            unreachable!()
        }
    }

    impl<T> Indexed for FrozenRecord<T> {
        #[inline]
        fn index(&self) -> usize {
            self.index
        }
    }

    impl<T: Copy> From<T> for Record<'_, T> {
        fn from(value: T) -> Self {
            Self::constant(value)
        }
    }

    impl<'a, T: Copy> Record<'a, T> {
        /**
         * Creates a new Record from a reference to an existing Record by applying
         * some unary function to it which operates on the type the Record wraps.
         *
         * To compute the new record, the unary function of some input x to some
         * output y is needed along with its derivative with respect to its input x.
         *
         * For example, tanh is a commonly used activation function, but the Real trait
         * does not include this operation and Record has no operations for it specifically.
         * However, you can use this function to compute the tanh of a Record like so:
         *
         * ```
         * use easy_ml::differentiation::{Record, WengertList};
         * let list = WengertList::new();
         * let x = Record::variable(0.7f32, &list);
         * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
         * // 1 / (cosh(x) * cosh(x))
         * let y = x.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
         * assert_eq!(y.derivatives()[&x], 1.0f32 / (0.7f32.cosh() * 0.7f32.cosh()));
         * ```
         */
        #[inline]
        pub fn unary(self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) -> Record<'a, T>
        where
            T: Zero + Clone,
        {
            match self.history {
                None => Record::constant(fx(self.number)),
                Some(history) => Record {
                    number: fx(self.number.clone()),
                    history: Some(history),
                    index: history.append_unary(self.index, dfx_dx(self.number.clone())),
                },
            }
        }

        /**
         * Creates a new Record from a reference to two existing Records by applying
         * some binary function to them which operates on two arguments of the type
         * the Records wrap.
         *
         * To compute the new record, the binary function of some inputs x and y to some
         * output z is needed along with its derivative with respect to its first input x and
         * its derivative with respect to its second input y.
         *
         * For example, atan2 takes two arguments, but the Real trait
         * does not include this operation and Record has no operations for it specifically.
         * However, you can use this function to compute the atan2 of two Records like so:
         *
         * ```
         * use easy_ml::differentiation::{Record, WengertList};
         * let list = WengertList::new();
         * let x = Record::variable(3.0f32, &list);
         * let y = Record::variable(3.0f32, &list);
         * // the derivative of atan2 with respect to x is y/(x*x + y*y)
         * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdx
         * // the derivative of atan2 with respect to y is -x/(x*x + y*y)
         * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdy
         * let z = x.binary(&y,
         *     |x, y| x.atan2(y),
         *     |x, y| y/((x*x) + (y*y)),
         *     |x, y| -x/((x*x) + (y*y))
         * );
         * let derivatives = z.derivatives();
         * let dx = derivatives[&x];
         * let dy = derivatives[&y];
         * ```
         */
        #[inline]
        #[track_caller]
        pub fn binary(
            self,
            rhs: Record<'a, T>,
            fxy: impl Fn(T, T) -> T,
            dfxy_dx: impl Fn(T, T) -> T,
            dfxy_dy: impl Fn(T, T) -> T,
        ) -> Record<'a, T>
        where
            T: Zero + Clone,
        {
            debug_assert!(
                same_list(&self, &rhs),
                "Records must be using the same WengertList"
            );
            match (self.history, rhs.history) {
                (None, None) => Record::constant(fxy(self.number, rhs.number)),
                (Some(history), None) => Record {
                    number: fxy(self.number.clone(), rhs.number.clone()),
                    history: Some(history),
                    index: history.append_unary(
                        // if rhs didn't have a history, don't track that derivative
                        self.index,
                        dfxy_dx(self.number.clone(), rhs.number.clone()),
                    ),
                },
                (None, Some(history)) => Record {
                    number: fxy(self.number.clone(), rhs.number.clone()),
                    history: Some(history),
                    index: history.append_unary(
                        // if self didn't have a history, don't track that derivative
                        rhs.index,
                        dfxy_dy(self.number, rhs.number),
                    ),
                },
                (Some(history), Some(_)) => Record {
                    number: fxy(self.number.clone(), rhs.number.clone()),
                    history: Some(history),
                    index: history.append_binary(
                        self.index,
                        dfxy_dx(self.number.clone(), rhs.number.clone()),
                        rhs.index,
                        dfxy_dy(self.number, rhs.number),
                    ),
                },
            }
        }
    }

    impl<T: Copy> Clone for Record<'_, T> {
        fn clone(&self) -> Self {
            Self {
                number: self.number.clone(),
                history: self.history,
                index: self.index,
            }
        }
    }
}

/**
 * Methods for appending Operations after borrowing the Wengert list.
 */
impl<'a, T: Copy> BorrowedWengertList<'a, T> {
    fn new(slot: &'a mut MaybeUninit<Operation<T>>) -> BorrowedWengertList<'a, T> {
        BorrowedWengertList { slot }
    }

    /**
     * Adds a value to the list which does not have any parent values.
     */
    #[inline]
    fn append_nullary(&mut self) -> Index<T>
    where
        T: Zero,
    {
        self.slot.write(Operation {
            // this index of the child is used for both indexes as these
            // won't be needed but will always be valid (ie point to a
            // real entry on the list)
            left_parent: std::ptr::null(),
            right_parent: std::ptr::null(),
            // for the parents 0 is used to zero out these calculations
            // as there are no parents
            left_derivative: T::zero(),
            right_derivative: T::zero(),
        });
        self.slot.as_ptr()
    }

    /**
     * Adds a value to the list which has one parent.
     *
     * For an output w_N which depends on one parent w_N-1
     * the derivative cached here is δw_N / δw_N-1
     *
     * For example, if z = sin(x), then δz/δx = cos(x)
     */
    #[inline]
    fn append_unary(&mut self, parent: Index<T>, derivative: T) -> Index<T>
    where
        T: Zero,
    {
        std::ptr::from_ref(self.slot.write(Operation {
            left_parent: parent,
            // this index of the child is used as this index won't be needed
            // but will always be valid (ie points to a real entry on the list)
            right_parent: std::ptr::null(),
            left_derivative: derivative,
            // for the right parent 0 is used to zero out this calculation
            // as there is no right parent
            right_derivative: T::zero(),
        }))
    }

    /**
     * Adds a value to the list which has two parents.
     *
     * For an output w_N which depends on two parents w_N-1
     * and w_N-2 the derivatives cached here are δw_N / δw_N-1
     * and δw_N / δw_N-2.
     *
     * For example, if z = y + x, then δz/δy = 1 and δz/δx = 1
     * For example, if z = y * x, then δz/δy = x and δz/δ/x = y
     */
    #[inline]
    fn append_binary(
        &mut self,
        left_parent: Index<T>,
        left_derivative: T,
        right_parent: Index<T>,
        right_derivative: T,
    ) -> Index<T> {
        std::ptr::from_ref(self.slot.write(Operation {
            left_parent,
            right_parent,
            left_derivative,
            right_derivative,
        }))
    }
}

impl<T: Copy> Slot<T> {
    #[inline(always)]
    fn with_mut<R>(&self, f: impl FnOnce(&mut MaybeUninit<Operation<T>>) -> R) -> R {
        let ptr = self.0.get();
        f(unsafe { ptr.as_mut().unwrap_unchecked() })
    }

    #[inline(always)]
    fn as_ptr(&self) -> Index<T> {
        self.0.get() as *const _ as Index<T>
    }
}
